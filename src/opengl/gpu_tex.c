/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "gpu.h"
#include "formats.h"
#include "utils.h"

#ifdef PL_HAVE_UNIX
#include <unistd.h>
#include <errno.h>
#endif

static bool yuv_compare_owner(struct gl_yuv_compare *c);

// Independent session: target is borrowed only during the one render. The
// large diagnostic arrays deliberately use fallible malloc, not aborting pl_alloc.
struct pl_opengl_uv_atlas_anchor_t {
    struct pl_uv_atlas_anchor dispatch;
    bool production_seed;
    pl_gpu gpu;
    pl_tex target;
    struct pl_opengl_uv_atlas_anchor_result result;
    EGLDisplay owner_display;
    EGLContext owner_context;
    EGLSurface owner_draw, owner_read;
    GLuint fbo, attachment;
    GLenum attachment_type;
    uint64_t entry_epoch;
    GLsync fence;
    uint32_t *rgb, *atlas_cpu;
    char *a0_source, *atlas_source, *vertex_source, *original_body;
    struct pl_opengl_yuv_compare_quad_layout layout;
    uint8_t *vertices;
    bool consumed, finished, invalidated, baseline_ok, completed, transaction_hold;
    bool has_material;
    struct pl_opengl_uv_atlas_material material;
    struct pl_opengl_uv_atlas_read_t *reader;
    uintptr_t input_texture_token;
    // Seeding pass's four corner UV values (the actual mapping numerics the
    // archived table is valid for); the consumer gate byte-compares these.
    float seed_uv[8];
    bool has_seed_uv;
    pl_tex input_watch; // watched until ordinal1 validates; never pins the texture
};

struct pl_opengl_uv_atlas_cache_t {
    pl_opengl_uv_atlas_anchor seed; // detached, owns immutable atlas + provenance
    struct pl_opengl_uv_atlas_cache_info info;
    struct pl_opengl_uv_atlas_read_t *active;
    unsigned int completed_reads;
    bool invalidated;
};
struct pl_opengl_uv_atlas_read_t {
    pl_opengl_uv_atlas_anchor state;
    pl_opengl_uv_atlas_cache cache;
    struct pl_opengl_uv_atlas_read_result result;
    uint32_t *uv, *half;
    char *b0_source;
};
#define MKS_ATLAS_NO_TRANSFORM
static bool uv_atlas_sealed(pl_opengl_uv_atlas_anchor c);
#include "../atlas_anchor.h"

void gl_tex_destroy(pl_gpu gpu, pl_tex tex)
{
    struct pl_tex_gl *detached = PL_PRIV(tex);
    if (detached->atlas_input_anchor) {
        // The watched input texture is renderer-internal and routinely freed
        // at frame end. Losing it must not invalidate the archived seed: the
        // consumer re-checks the recorded input token on every pass instead.
        detached->atlas_input_anchor->input_watch=NULL;
        detached->atlas_input_anchor=NULL;
    }
    if (detached->atlas_read_owner) {
        detached->atlas_read_owner->state->dispatch.yuv_target=NULL;
        detached->atlas_read_owner->state->invalidated=true;
        detached->atlas_read_owner->cache->invalidated=true;
        detached->atlas_read_owner=NULL;
    }
    if (detached->atlas_anchor) {
        if (detached->atlas_anchor->reader) detached->atlas_anchor->reader->cache->invalidated=true;
        pl_opengl_uv_atlas_anchor a = detached->atlas_anchor;
        // The production seed transaction finishes synchronously inside its
        // seeding frame, before this swapchain FBO can be handed back and
        // recycled on a later frame. Once sealed, the UV table lives in its
        // own storage buffer and poll/promote no longer touch the target, so
        // losing the recycled FBO detaches silently. An unfinished seed (or
        // any diagnostic anchor) still keeps the strict contract: losing the
        // transaction's render target invalidates it.
        if (!uv_atlas_sealed(a)) {
            PL_ERR(a->gpu, "MKS-ANCHOR-DIAG target-destroy invalidate prod=%d finished=%d baseline=%d",
                   a->production_seed ? 1 : 0, a->finished ? 1 : 0, a->baseline_ok ? 1 : 0);
            a->invalidated = true;
        }
        a->target = NULL;
        detached->atlas_anchor = NULL;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "Failed uninitializing texture, leaking resources!");
        return;
    }

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (tex_gl->compare && !yuv_compare_owner(tex_gl->compare)) {
        // A foreign context may reuse the same GL names. Retain the whole
        // diagnostic wrapper rather than deleting another context's objects.
        PL_ERR(gpu, "Foreign EGL owner during compare target destroy, leaking resources!");
        RELEASE_CURRENT();
        return;
    }
    if (tex_gl->compare)
        pl_opengl_yuv_compare_cancel(gpu, tex);
    if (tex_gl->fbo && !tex_gl->wrapped_fb)
        gl->DeleteFramebuffers(1, &tex_gl->fbo);
    if (tex_gl->image) {
        struct pl_gl *p = PL_PRIV(gpu);
        eglDestroyImageKHR(p->egl_dpy, tex_gl->image);
    }
    if (!tex_gl->wrapped_tex)
        gl->DeleteTextures(1, &tex_gl->texture);

#ifdef PL_HAVE_UNIX
    if (tex_gl->fd != -1)
        close(tex_gl->fd);
#endif

    gl_check_err(gpu, "gl_tex_destroy");
    RELEASE_CURRENT();
    pl_free((void *) tex);
}

static GLbitfield tex_barrier(pl_tex tex)
{
    GLbitfield barrier = 0;
    const struct pl_tex_params *params = &tex->params;

    if (params->sampleable)
        barrier |= GL_TEXTURE_FETCH_BARRIER_BIT;
    if (params->renderable || params->blit_src || params->blit_dst)
        barrier |= GL_FRAMEBUFFER_BARRIER_BIT;
    if (params->storable)
        barrier |= GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
    if (params->host_writable || params->host_readable)
        barrier |= GL_TEXTURE_UPDATE_BARRIER_BIT;

    return barrier;
}

#define ADD_ATTRIB(name, value)                                     \
    do {                                                            \
        assert(num_attribs + 3 < PL_ARRAY_SIZE(attribs));           \
        attribs[num_attribs++] = (name);                            \
        attribs[num_attribs++] = (value);                           \
    } while (0)

#define ADD_DMABUF_PLANE_ATTRIBS(plane, fd, offset, stride)         \
    do {                                                            \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _FD_EXT,           \
                   fd);                                             \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _OFFSET_EXT,       \
                   offset);                                         \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _PITCH_EXT,        \
                   stride);                                         \
    } while (0)

#define ADD_DMABUF_PLANE_MODIFIERS(plane, mod)                      \
    do {                                                            \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _MODIFIER_LO_EXT,  \
                   (uint32_t) ((mod) & 0xFFFFFFFFlu));              \
        ADD_ATTRIB(EGL_DMA_BUF_PLANE ## plane ## _MODIFIER_HI_EXT,  \
                   (uint32_t) (((mod) >> 32u) & 0xFFFFFFFFlu));     \
    } while (0)

static bool gl_tex_import(pl_gpu gpu,
                          enum pl_handle_type handle_type,
                          const struct pl_shared_mem *shared_mem,
                          struct pl_tex_t *tex)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    if (!MAKE_CURRENT())
        return false;

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    const struct pl_tex_params *params = &tex->params;

    int attribs[20] = {};
    int num_attribs = 0;
    ADD_ATTRIB(EGL_WIDTH,  params->w);
    ADD_ATTRIB(EGL_HEIGHT, params->h);

    switch (handle_type) {

#ifdef PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        if (shared_mem->handle.fd == -1) {
            PL_ERR(gpu, "%s: invalid fd", __func__);
            goto error;
        }

        tex_gl->fd = dup(shared_mem->handle.fd);
        if (tex_gl->fd == -1) {
            PL_ERR(gpu, "%s: cannot duplicate fd %d for importing: %s",
                   __func__, shared_mem->handle.fd, strerror(errno));
            goto error;
        }

        ADD_ATTRIB(EGL_LINUX_DRM_FOURCC_EXT, params->format->fourcc);
        ADD_DMABUF_PLANE_ATTRIBS(0, tex_gl->fd, shared_mem->offset,
                                 PL_DEF(shared_mem->stride_w, params->w));
        if (p->has_modifiers)
            ADD_DMABUF_PLANE_MODIFIERS(0, shared_mem->drm_format_mod);

        attribs[num_attribs] = EGL_NONE;

        // EGL_LINUX_DMA_BUF_EXT requires EGL_NO_CONTEXT
        tex_gl->image = eglCreateImageKHR(p->egl_dpy,
                                          EGL_NO_CONTEXT,
                                          EGL_LINUX_DMA_BUF_EXT,
                                          (EGLClientBuffer) NULL,
                                          attribs);

        break;
#else // !PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        pl_unreachable();
#endif

    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_HOST_PTR:
    case PL_HANDLE_FD:
    case PL_HANDLE_MTL_TEX:
    case PL_HANDLE_IOSURFACE:
        pl_unreachable();

    }

    if (!egl_check_err(gpu, "eglCreateImageKHR") || !tex_gl->image)
        goto error;

    // tex_gl->image should be already bound
    if (p->has_egl_storage) {
        gl->EGLImageTargetTexStorageEXT(GL_TEXTURE_2D, tex_gl->image, NULL);
    } else {
        gl->EGLImageTargetTexture2DOES(GL_TEXTURE_2D, tex_gl->image);
    }
    if (!egl_check_err(gpu, "EGLImageTargetTexture2DOES"))
        goto error;

    RELEASE_CURRENT();
    return true;

error:
    PL_ERR(gpu, "Failed importing GL texture!");
    RELEASE_CURRENT();
    return false;
}

static EGLenum egl_from_gl_target(pl_gpu gpu, int target)
{
    switch(target) {
    case GL_TEXTURE_2D: return EGL_GL_TEXTURE_2D;
    case GL_TEXTURE_3D: return EGL_GL_TEXTURE_3D;
    default:
        PL_ERR(gpu, "%s: unsupported texture target 0x%x", __func__, target);
        return 0;
    }
}

static bool gl_tex_export(pl_gpu gpu, enum pl_handle_type handle_type,
                          bool preserved, struct pl_tex_t *tex)
{
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_gl *p = PL_PRIV(gpu);

    EGLenum egltarget = egl_from_gl_target(gpu, tex_gl->target);
    if (!egltarget)
        goto error;

    int attribs[] = {
        EGL_IMAGE_PRESERVED, preserved,
        EGL_NONE,
    };

    // We assume that tex_gl->texture is already bound
    tex_gl->image = eglCreateImageKHR(p->egl_dpy,
                                      p->egl_ctx,
                                      egltarget,
                                      (EGLClientBuffer) (uintptr_t) tex_gl->texture,
                                      attribs);
    if (!egl_check_err(gpu, "eglCreateImageKHR") || !tex_gl->image)
        goto error;

    switch (handle_type) {

#ifdef PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF: {
        int fourcc = 0;
        int num_planes = 0;
        EGLuint64KHR modifier = 0;
        bool ok;
        ok = eglExportDMABUFImageQueryMESA(p->egl_dpy,
                                           tex_gl->image,
                                           &fourcc,
                                           &num_planes,
                                           &modifier);
        if (!egl_check_err(gpu, "eglExportDMABUFImageQueryMESA") || !ok)
            goto error;

        if (fourcc != tex->params.format->fourcc) {
            PL_ERR(gpu, "Exported DRM format %s does not match fourcc of "
                   "specified pl_fmt %s? Please open a bug.",
                   PRINT_FOURCC(fourcc), PRINT_FOURCC(tex->params.format->fourcc));
            goto error;
        }

        if (num_planes != 1) {
            PL_ERR(gpu, "Unsupported number of planes: %d", num_planes);
            goto error;
        }

        int offset = 0, stride = 0;
        ok = eglExportDMABUFImageMESA(p->egl_dpy,
                                      tex_gl->image,
                                      &tex_gl->fd,
                                      &stride,
                                      &offset);
        if (!egl_check_err(gpu, "eglExportDMABUFImageMesa") || !ok)
            goto error;

        off_t fdsize = lseek(tex_gl->fd, 0, SEEK_END);
        off_t err = fdsize > 0 && lseek(tex_gl->fd, 0, SEEK_SET);
        if (fdsize <= 0 || err < 0) {
            PL_ERR(gpu, "Failed querying FD size: %s", strerror(errno));
            goto error;
        }

        tex->shared_mem = (struct pl_shared_mem) {
            .handle.fd = tex_gl->fd,
            .size = fdsize,
            .offset = offset,
            .drm_format_mod = modifier,
            .stride_w = stride,
        };
        break;
    }
#else // !PL_HAVE_UNIX
    case PL_HANDLE_DMA_BUF:
        pl_unreachable();
#endif

    case PL_HANDLE_WIN32:
    case PL_HANDLE_WIN32_KMT:
    case PL_HANDLE_HOST_PTR:
    case PL_HANDLE_FD:
    case PL_HANDLE_MTL_TEX:
    case PL_HANDLE_IOSURFACE:
        pl_unreachable();

    }

    return true;

error:
    PL_ERR(gpu, "Failed exporting GL texture!");
    return false;
}

static const char *fb_err_str(GLenum err)
{
    switch (err) {
#define CASE(name) case name: return #name
    CASE(GL_FRAMEBUFFER_COMPLETE);
    CASE(GL_FRAMEBUFFER_UNDEFINED);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
    CASE(GL_FRAMEBUFFER_UNSUPPORTED);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
    CASE(GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS);
#undef CASE

    default: return "unknown error";
    }
}

pl_tex gl_tex_create(pl_gpu gpu, const struct pl_tex_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_t *tex = pl_zalloc_obj(NULL, tex, struct pl_tex_gl);
    tex->params = *params;
    tex->params.initial_data = NULL;
    tex->sampler_type = PL_SAMPLER_NORMAL;

    struct pl_tex_gl *tex_gl = PL_PRIV(tex);

    const struct gl_format **fmtp = PL_PRIV(params->format);
    const struct gl_format *fmt = *fmtp;
    *tex_gl = (struct pl_tex_gl) {
        .format = fmt->fmt,
        .iformat = fmt->ifmt,
        .type = fmt->type,
        .barrier = tex_barrier(tex),
        .fd = -1,
    };

    static const GLint targets[] = {
        [1] = GL_TEXTURE_1D,
        [2] = GL_TEXTURE_2D,
        [3] = GL_TEXTURE_3D,
    };

    int dims = pl_tex_params_dimension(*params);
    pl_assert(dims >= 1 && dims <= 3);
    tex_gl->target = targets[dims];

    gl->GenTextures(1, &tex_gl->texture);
    gl->BindTexture(tex_gl->target, tex_gl->texture);

    if (params->import_handle) {
        if (!gl_tex_import(gpu, params->import_handle, &params->shared_mem, tex))
            goto error;
    } else {
        gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Use immutable storage for storable textures. This works around
        // an NVIDIA OpenGL driver bug. With mutable-storage textures the
        // driver defers allocation until first use, and the first imageStore
        // from a compute shader is silently dropped despite a correct
        // GL_SHADER_IMAGE_ACCESS_BARRIER_BIT / GL_TEXTURE_FETCH_BARRIER_BIT
        // barriers.
        switch (dims) {
        case 1:
            if (params->storable) {
                gl->TexStorage1D(tex_gl->target, 1, tex_gl->iformat, params->w);
                if (params->initial_data) {
                    gl->TexSubImage1D(tex_gl->target, 0, 0, params->w,
                                      tex_gl->format, tex_gl->type,
                                      params->initial_data);
                }
            } else {
                gl->TexImage1D(tex_gl->target, 0, tex_gl->iformat, params->w, 0,
                               tex_gl->format, tex_gl->type, params->initial_data);
            }
            break;
        case 2:
            if (params->storable) {
                gl->TexStorage2D(tex_gl->target, 1, tex_gl->iformat,
                                 params->w, params->h);
                if (params->initial_data) {
                    gl->TexSubImage2D(tex_gl->target, 0, 0, 0,
                                      params->w, params->h,
                                      tex_gl->format, tex_gl->type,
                                      params->initial_data);
                }
            } else {
                gl->TexImage2D(tex_gl->target, 0, tex_gl->iformat,
                               params->w, params->h, 0,
                               tex_gl->format, tex_gl->type, params->initial_data);
            }
            break;
        case 3:
            if (params->storable) {
                gl->TexStorage3D(tex_gl->target, 1, tex_gl->iformat,
                                 params->w, params->h, params->d);
                if (params->initial_data) {
                    gl->TexSubImage3D(tex_gl->target, 0, 0, 0, 0,
                                      params->w, params->h, params->d,
                                      tex_gl->format, tex_gl->type,
                                      params->initial_data);
                }
            } else {
                gl->TexImage3D(tex_gl->target, 0, tex_gl->iformat,
                               params->w, params->h, params->d, 0,
                               tex_gl->format, tex_gl->type, params->initial_data);
            }
            break;
        }

        gl->PixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    if (params->export_handle) {
        if (!gl_tex_export(gpu, params->export_handle, params->initial_data, tex))
            goto error;
    }

    gl->BindTexture(tex_gl->target, 0);

    if (!gl_check_err(gpu, "gl_tex_create: texture"))
        goto error;

    bool need_fbo = tex->params.renderable;
    if (tex->params.blit_src || tex->params.blit_dst) {
        if (dims != 2) {
            PL_ERR(gpu, "Blittable textures may only be 2D!");
            goto error;
        }

        need_fbo = true;
    }

    bool can_fbo = tex->params.format->caps & PL_FMT_CAP_RENDERABLE &&
                   tex->params.d == 0;

    // Try creating an FBO for host-readable textures, since this allows
    // reading back with glReadPixels instead of glGetTexImage. (Additionally,
    // GLES does not support glGetTexImage)
    if (tex->params.host_readable && (can_fbo || p->gles_ver))
        need_fbo = true;

    if (need_fbo) {
        if (!can_fbo) {
            PL_ERR(gpu, "Trying to create a renderable/blittable/readable "
                   "texture with an incompatible (non-renderable) format!");
            goto error;
        }

        const GLenum target = p->gles_ver && p->gles_ver < 30 ?
            GL_FRAMEBUFFER : GL_READ_FRAMEBUFFER;

        gl->GenFramebuffers(1, &tex_gl->fbo);
        gl->BindFramebuffer(target, tex_gl->fbo);
        switch (dims) {
        case 1:
            gl->FramebufferTexture1D(target, GL_COLOR_ATTACHMENT0,
                                     GL_TEXTURE_1D, tex_gl->texture, 0);
            break;
        case 2:
            gl->FramebufferTexture2D(target, GL_COLOR_ATTACHMENT0,
                                     GL_TEXTURE_2D, tex_gl->texture, 0);
            break;
        case 3: pl_unreachable();
        }

        GLenum err = gl->CheckFramebufferStatus(target);
        if (err != GL_FRAMEBUFFER_COMPLETE) {
            gl->BindFramebuffer(target, 0);
            PL_ERR(gpu, "Failed creating framebuffer: %s", fb_err_str(err));
            goto error;
        }

        if (params->host_readable && p->gles_ver) {
            GLint read_type = 0, read_fmt = 0;
            gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
            gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
            if (read_type != tex_gl->type || read_fmt != tex_gl->format) {
                gl->BindFramebuffer(target, 0);
                PL_ERR(gpu, "Trying to create host_readable texture whose "
                       "implementation-defined pixel read format "
                       "(type=0x%X, fmt=0x%X) does not match the texture's "
                       "internal format (type=0x%X, fmt=0x%X)! This is a "
                       "GLES/driver limitation, there's little we can do "
                       "about it.",
                       read_type, read_fmt, tex_gl->type, tex_gl->format);
                goto error;
            }
        }

        gl->BindFramebuffer(target, 0);
        if (!gl_check_err(gpu, "gl_tex_create: fbo"))
            goto error;
    }

    RELEASE_CURRENT();
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    RELEASE_CURRENT();
    return NULL;
}

static bool gl_fb_query(pl_gpu gpu, int fbo, struct pl_fmt_t *fmt,
                        struct gl_format *glfmt)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    *fmt = (struct pl_fmt_t) {
        .name = "fbo",
        .type = PL_FMT_UNKNOWN,
        .caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLENDABLE,
        .num_components = 4,
        .component_depth = {8, 8, 8, 8}, // default to rgba8
        .sample_order = {0, 1, 2, 3},
    };

    *glfmt = (struct gl_format) {
        .fmt = GL_RGBA,
    };

    bool can_query = gl_test_ext(gpu, "GL_ARB_framebuffer_object", 30, 20);
    if (!fbo && p->gles_ver && p->gles_ver < 30)
        can_query = false; // can't query default framebuffer on GLES 2.0

    if (can_query) {
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

        GLenum obj = p->gles_ver ? GL_BACK : GL_BACK_LEFT;
        if (fbo != 0)
            obj = GL_COLOR_ATTACHMENT0;

        GLint type = 0;
        gl->GetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                            GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &type);
        switch (type) {
        case GL_FLOAT:                  fmt->type = PL_FMT_FLOAT; break;
        case GL_INT:                    fmt->type = PL_FMT_SINT; break;
        case GL_UNSIGNED_INT:           fmt->type = PL_FMT_UINT; break;
        case GL_SIGNED_NORMALIZED:      fmt->type = PL_FMT_SNORM; break;
        case GL_UNSIGNED_NORMALIZED:    fmt->type = PL_FMT_UNORM; break;
        default:                        fmt->type = PL_FMT_UNKNOWN; break;
        }

        gl->GetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &fmt->component_depth[0]);
        gl->GetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &fmt->component_depth[1]);
        gl->GetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &fmt->component_depth[2]);
        gl->GetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, obj,
                GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE, &fmt->component_depth[3]);

        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        gl_check_err(gpu, "gl_fb_query");

        if (!fmt->component_depth[0]) {
            PL_INFO(gpu, "OpenGL framebuffer did not export depth information,"
                    "assuming 8-bit framebuffer");
            for (int i = 0; i < PL_ARRAY_SIZE(fmt->component_depth); i++)
                fmt->component_depth[i] = 8;
        }

        // Strip missing components from component map
        while (!fmt->component_depth[fmt->num_components - 1]) {
            fmt->num_components--;
            pl_assert(fmt->num_components);
        }
    }

    int gpu_bits = 0;
    for (int i = 0; i < 4; i++)
        gpu_bits += fmt->component_depth[i];
    fmt->internal_size = (gpu_bits + 7) / 8;

    size_t host_size = 0;
    switch (fmt->type) {
    case PL_FMT_UNKNOWN:
        fmt->opaque = true;
        return true;
    case PL_FMT_FLOAT:
        glfmt->type = GL_FLOAT;
        host_size = sizeof(float);
        break;
    case PL_FMT_UNORM:
    case PL_FMT_UINT:
        if (gpu_bits > 32) {
            glfmt->type = GL_UNSIGNED_SHORT;
            host_size = sizeof(uint16_t);
        } else {
            glfmt->type = GL_UNSIGNED_BYTE;
            host_size = sizeof(uint8_t);
        }
        break;
    case PL_FMT_SNORM:
    case PL_FMT_SINT:
        if (gpu_bits > 32) {
            glfmt->type = GL_SHORT;
            host_size = sizeof(int16_t);
        } else {
            glfmt->type = GL_BYTE;
            host_size = sizeof(int8_t);
        }
        break;
    case PL_FMT_TYPE_COUNT:
        pl_unreachable();
    }

    fmt->texel_size = fmt->num_components * host_size;
    for (int i = 0; i < fmt->num_components; i++)
        fmt->host_bits[i] = 8 * host_size;
    fmt->caps |= PL_FMT_CAP_HOST_READABLE;
    if (!p->gles_ver || p->gles_ver >= 30)
        fmt->caps |= PL_FMT_CAP_BLITTABLE;

    return true;
}

pl_tex pl_opengl_wrap(pl_gpu gpu, const struct pl_opengl_wrap_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return NULL;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_t *tex = pl_alloc_obj(NULL, tex, struct pl_tex_gl);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    *tex = (struct pl_tex_t) {
        .params = {
            .w = params->width,
            .h = params->height,
            .d = params->depth,
        },
    };

    pl_fmt fmt = NULL;
    const struct gl_format *glfmt = NULL;

    bool yuv_output = params->output_encoding ==
                      PL_OPENGL_OUTPUT_RGB16F_TO_BT2020_FULL_YUV;
    if (params->output_encoding != PL_OPENGL_OUTPUT_NORMAL && !yuv_output) {
        PL_ERR(gpu, "Unknown output encoding");
        goto error;
    }
    if (yuv_output && (params->texture || params->depth ||
                      params->sampler_type != PL_SAMPLER_NORMAL ||
                      params->width <= 0 || params->height <= 0 ||
                      params->iformat != GL_RGBA16F || p->gles_ver < 31 ||
                      gpu->glsl.version < 310 ||
                      !pl_opengl_has_ext(p->gl, "GL_EXT_YUV_target")))
    {
        PL_ERR(gpu, "Unsupported RGB16F to YUV drawable output contract");
        goto error;
    }
    if (params->texture || yuv_output) {
        // Wrapping texture: Require matching iformat
        pl_assert(params->iformat);
        for (int i = 0; i < gpu->num_formats; i++) {
            const struct gl_format **glfmtp = PL_PRIV(gpu->formats[i]);
            if ((*glfmtp)->ifmt == params->iformat) {
                fmt = gpu->formats[i];
                glfmt = *glfmtp;
                break;
            }
        }

        if (!fmt) {
            PL_ERR(gpu, "Failed mapping iformat %d to any equivalent `pl_fmt`",
                   params->iformat);
            goto error;
        }
    } else {
        // Wrapping framebuffer: Allocate/infer generic FBO format
        fmt = pl_alloc_obj((void *) gpu, fmt, const struct gl_format *);
        glfmt = pl_alloc_ptr((void *) fmt, glfmt);
        const struct gl_format **glfmtp = PL_PRIV(fmt);
        *glfmtp = glfmt;
        if (!gl_fb_query(gpu, params->framebuffer,
                         (struct pl_fmt_t *) fmt,
                         (struct gl_format *) glfmt))
        {
            PL_ERR(gpu, "Failed querying framebuffer specifics!");
            pl_free((void *) fmt);
            goto error;
        }
    }

    *tex_gl = (struct pl_tex_gl) {
        .rgb16f_to_yuv = yuv_output,
        .target = params->target,
        .texture = params->texture,
        .fbo = params->framebuffer,
        .wrapped_tex = !!params->texture,
        .wrapped_fb = params->framebuffer || !params->texture,
        .iformat = glfmt->ifmt,
        .format = glfmt->fmt,
        .type = glfmt->type,
        .fd = -1,
    };

    int dims = pl_tex_params_dimension(tex->params);
    if (!tex_gl->target) {
        switch (dims) {
        case 1: tex_gl->target = GL_TEXTURE_1D; break;
        case 2: tex_gl->target = GL_TEXTURE_2D; break;
        case 3: tex_gl->target = GL_TEXTURE_3D; break;
        }
    }

    // Map texture-specific sampling metadata
    if (params->texture) {
        switch (params->target) {
        case GL_TEXTURE_1D:
            if (params->width || params->depth) {
                PL_ERR(gpu, "Invalid texture dimensions for GL_TEXTURE_1D");
                goto error;
            }
            // fall through
        case GL_TEXTURE_2D:
            if (params->depth) {
                PL_ERR(gpu, "Invalid texture dimensions for GL_TEXTURE_2D");
                goto error;
            }
            // fall through
        case 0:
        case GL_TEXTURE_3D:
            tex->sampler_type = PL_SAMPLER_NORMAL;
            break;

        case GL_TEXTURE_RECTANGLE: tex->sampler_type = PL_SAMPLER_RECT; break;
        case GL_TEXTURE_EXTERNAL_OES: tex->sampler_type = PL_SAMPLER_EXTERNAL; break;

        default:
            PL_ERR(gpu, "Failed mapping texture target %u to any equivalent "
                   "`pl_sampler_type`", params->target);
            goto error;
        }
        if (params->sampler_type != PL_SAMPLER_NORMAL) {
            if (params->target != GL_TEXTURE_EXTERNAL_OES ||
                params->sampler_type != PL_SAMPLER_EXTERNAL_YUV)
            {
                PL_ERR(gpu, "Unsupported explicit sampler type for texture target");
                goto error;
            }
            if (p->gles_ver < 30 ||
                !pl_opengl_has_ext(p->gl, "GL_EXT_YUV_target"))
            {
                PL_ERR(gpu, "Requested PL_SAMPLER_EXTERNAL_YUV but "
                       "GL_EXT_YUV_target (GLES 3.0+) is unavailable");
                goto error;
            }
            tex->sampler_type = params->sampler_type;
        }
    }

    // Create optional extra fbo if needed/possible
    bool can_fbo = tex_gl->texture &&
                   (fmt->caps & PL_FMT_CAP_RENDERABLE) &&
                   tex->sampler_type != PL_SAMPLER_EXTERNAL &&
                   tex->sampler_type != PL_SAMPLER_EXTERNAL_YUV &&
                   dims < 3;

    if (can_fbo && !tex_gl->fbo) {
        const GLenum target = p->gles_ver && p->gles_ver < 30 ?
            GL_FRAMEBUFFER : GL_READ_FRAMEBUFFER;

        gl->GenFramebuffers(1, &tex_gl->fbo);
        gl->BindFramebuffer(target, tex_gl->fbo);
        switch (dims) {
        case 1:
            gl->FramebufferTexture1D(target, GL_COLOR_ATTACHMENT0,
                                     tex_gl->target, tex_gl->texture, 0);
            break;
        case 2:
            gl->FramebufferTexture2D(target, GL_COLOR_ATTACHMENT0,
                                     tex_gl->target, tex_gl->texture, 0);
            break;
        }

        GLenum err = gl->CheckFramebufferStatus(target);
        if (err != GL_FRAMEBUFFER_COMPLETE) {
            gl->BindFramebuffer(target, 0);
            PL_ERR(gpu, "Failed creating framebuffer: error code %d", err);
            goto error;
        }

        if (p->gles_ver) {
            GLint read_type = 0, read_fmt = 0;
            gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
            gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
            tex->params.host_readable = read_type == tex_gl->type &&
                                        read_fmt == tex_gl->format;
        } else {
            tex->params.host_readable = true;
        }

        gl->BindFramebuffer(target, 0);
        if (!gl_check_err(gpu, "pl_opengl_wrap: fbo"))
            goto error;
    }

    // Complete the process of inferring the texture capabilities
    tex->params.format = fmt;
    if (tex_gl->texture) {
        tex->params.sampleable = fmt->caps & PL_FMT_CAP_SAMPLEABLE;
        tex->params.storable = fmt->caps & PL_FMT_CAP_STORABLE;
        tex->params.host_writable = !fmt->opaque;
        tex->params.host_readable |= fmt->caps & PL_FMT_CAP_HOST_READABLE;
    }
    if (tex_gl->fbo || tex_gl->wrapped_fb) {
        tex->params.renderable = fmt->caps & PL_FMT_CAP_RENDERABLE;
        tex->params.host_readable |= fmt->caps & PL_FMT_CAP_HOST_READABLE;
        if (dims == 2 && (fmt->caps & PL_FMT_CAP_BLITTABLE)) {
            tex->params.blit_src = true;
            tex->params.blit_dst = true;
        }
    }

    if (yuv_output) {
        // Deliberately retain logical RGB16F precision for renderer inference,
        // but expose none of the operations invalid for the physical drawable.
        tex->params.sampleable = false;
        tex->params.storable = false;
        tex->params.blit_src = tex->params.blit_dst = false;
        tex->params.host_readable = tex->params.host_writable = false;
    }
    tex_gl->barrier = tex_barrier(tex);
    RELEASE_CURRENT();
    return tex;

error:
    gl_tex_destroy(gpu, tex);
    RELEASE_CURRENT();
    return NULL;
}

// These EGL queries never issue GL commands. MAKE_CURRENT may only lock when
// the caller supplied no context callback, so it cannot establish ownership.
static bool yuv_compare_owner(struct gl_yuv_compare *c)
{
    bool ok = eglGetCurrentDisplay && eglGetCurrentContext && eglGetCurrentSurface &&
              eglGetCurrentDisplay() == c->owner_display &&
              eglGetCurrentContext() == c->owner_context &&
              eglGetCurrentSurface(EGL_DRAW) == c->owner_draw &&
              eglGetCurrentSurface(EGL_READ) == c->owner_read;
    if (!ok)
        c->invalidated = true;
    return ok;
}

// Called with the owner current. Drain pending errors even if another API has
// already consumed the error: its monotonic epoch still invalidates this A.
static bool yuv_compare_epoch(pl_gpu gpu, struct gl_yuv_compare *c, const char *fun)
{
    struct pl_gl *p = PL_PRIV(gpu);
    bool clean = gl_check_err(gpu, fun);
    if (!clean || (c->finished && p->error_epoch != c->finish_epoch))
        c->invalidated = true;
    return !c->invalidated;
}

// This explicit transaction is diagnostic only. Its fence/readback never runs
// for an ordinary texture, or for production YUV output without compare_arm.
void pl_opengl_yuv_compare_cancel(pl_gpu gpu, pl_tex target)
{
    if (!gpu || !target || !pl_opengl_get(gpu))
        return;
    struct pl_tex_gl *t = PL_PRIV(target);
    if (!t->compare)
        return;
    if (!MAKE_CURRENT()) {
        t->compare->invalidated = true;
        return;
    }
    struct gl_yuv_compare *c = t->compare;
    if (!yuv_compare_owner(c)) {
        // Keep resources for an explicit retry with the original owner current.
        RELEASE_CURRENT();
        return;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (c->fence)
        gl->DeleteSync(c->fence);
    pl_buf_destroy(gpu, &c->dispatch.storage);
    pl_buf_destroy(gpu, &c->dispatch.storage_repeat);
    pl_buf_destroy(gpu, &c->dispatch.probe_storage);
    pl_free_ptr(&t->compare);
    gl_check_err(gpu, "yuv_compare_cancel");
    RELEASE_CURRENT();
}

static bool yuv_compare_prepare(pl_gpu gpu, pl_tex target, int binding)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    if (binding < 0 || !t->compare)
        return false;
    if (!MAKE_CURRENT()) {
        t->compare->invalidated = true;
        return false;
    }
    if (!yuv_compare_owner(t->compare) || t->compare->invalidated) {
        RELEASE_CURRENT();
        return false;
    }
    struct pl_gl *p = PL_PRIV(gpu);
    uint64_t before = p->error_epoch;
    const gl_funcs *gl = gl_funcs_get(gpu);
    GLint max = 0, previous = -1, generic = -1;
    gl->GetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &max);
    gl->GetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &generic);
    if (binding < max && gl->GetIntegeri_v)
        gl->GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, binding, &previous);
    bool clean = gl_check_err(gpu, "yuv_compare_prepare");
    bool ok = previous == 0 && generic == 0 && clean && p->error_epoch == before;
    if (!ok)
        t->compare->invalidated = true;
    // Existing generic pass cleanup unbinds storage slots. Requiring the new
    // diagnostic slot to be empty avoids destroying an external range binding.
    RELEASE_CURRENT();
    return ok;
}

// Explicit A-to-B dependency: no CPU readback, only same-context SSBO visibility.
// Ownership, pending errors and consumed/callback errors all invalidate the pair.
static bool yuv_compare_barrier(pl_gpu gpu, pl_tex target, unsigned int pair)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.uv_oracle || !c->consumed || c->finished || pair >= c->result.repetitions || pair >= 2 ||
        c->result.uv_oracle_barriers != pair || !c->result.draw_ok[pair * 2])
        return false;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return false;
    }
    if (!yuv_compare_owner(c) || c->invalidated) {
        RELEASE_CURRENT();
        return false;
    }
    struct pl_gl *p = PL_PRIV(gpu);
    const gl_funcs *gl = gl_funcs_get(gpu);
    uint64_t before = p->error_epoch;
    bool clean = gl_check_err(gpu, "yuv_compare_barrier_entry");
    bool ok = clean && before == p->error_epoch && gl->MemoryBarrier;
    if (ok) {
        gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        clean = gl_check_err(gpu, "yuv_compare_barrier");
        ok = clean && before == p->error_epoch;
    }
    c->result.uv_oracle_barrier_epoch_before[pair] = before;
    c->result.uv_oracle_barrier_epoch_after[pair] = p->error_epoch;
    c->result.uv_oracle_barrier_ok[pair] = ok;
    if (ok)
        c->result.uv_oracle_barriers++;
    else
        c->invalidated = true;
    RELEASE_CURRENT();
    return ok;
}

static void yuv_compare_bodies(pl_gpu gpu, pl_tex target, const char *original,
                               const char *tap, uint64_t tap_hash)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.sample_tap || !original || !*original || !tap || !*tap)
        return;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return;
    }
    bool owner = yuv_compare_owner(c);
    RELEASE_CURRENT();
    if (!owner || c->invalidated)
        return;
    c->original_body = pl_strdup0(c, pl_str0(original));
    c->tap_body = pl_strdup0(c, pl_str0(tap));
    c->result.sample_tap_body_hash = tap_hash;
    c->result.sample_sites = 1;
}

bool pl_opengl_yuv_compare_bodies(pl_gpu gpu, pl_tex target,
                                 const char **original, const char **tap)
{
    if (!gpu || !target || !pl_opengl_get(gpu) || !original || !tap)
        return false;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.sample_tap || !c->original_body || !c->tap_body)
        return false;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return false;
    }
    bool owner = yuv_compare_owner(c);
    RELEASE_CURRENT();
    if (!owner || c->invalidated)
        return false;
    *original = c->original_body;
    *tap = c->tap_body;
    return true;
}

static void yuv_compare_vertices(pl_gpu gpu, pl_tex target, const char *candidate,
                                 const char *baseline, const char *uv)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.invariant_b || !candidate || !*candidate || !baseline || !*baseline ||
        !uv || !*uv || strlen(uv) >= sizeof(c->result.invariant_uv_identifier))
        return;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return;
    }
    bool owner = yuv_compare_owner(c);
    RELEASE_CURRENT();
    if (!owner || c->invalidated)
        return;
    if (!c->vertex_source || strcmp(c->vertex_source, baseline)) {
        c->invalidated = true;
        return;
    }
    c->candidate_vertex_source = pl_strdup0(c, pl_str0(candidate));
    memcpy(c->result.invariant_uv_identifier, uv, strlen(uv) + 1);
}

bool pl_opengl_yuv_compare_vertices(pl_gpu gpu, pl_tex target,
                                   const char **candidate, const char **baseline)
{
    if (!gpu || !target || !pl_opengl_get(gpu) || !candidate || !baseline)
        return false;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.invariant_b || !c->candidate_vertex_source || !c->vertex_source ||
        !c->result.invariant_uv_identifier[0])
        return false;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return false;
    }
    bool owner = yuv_compare_owner(c);
    RELEASE_CURRENT();
    if (!owner || c->invalidated)
        return false;
    *candidate = c->candidate_vertex_source;
    *baseline = c->vertex_source;
    return true;
}

// Capture-bound copies of the exact CPU vertex payloads supplied to A/B.
static bool yuv_compare_quad_payloads(pl_gpu gpu, pl_tex target,
    const uint8_t *baseline, const uint8_t *candidate,
    const struct pl_opengl_yuv_compare_quad_layout *layout)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.quad_order || !c->consumed || c->finished || c->quad_recorded ||
        !baseline || !candidate || !layout)
        return false;
    struct pl_gl *p = PL_PRIV(gpu);
    uint64_t before = p->error_epoch;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return false;
    }
    if (!yuv_compare_owner(c) || c->invalidated) {
        RELEASE_CURRENT();
        return false;
    }
    bool clean = gl_check_err(gpu, "yuv_compare_quad_payloads_entry");
    if (!clean || p->error_epoch != before) {
        c->invalidated = true;
        RELEASE_CURRENT();
        return false;
    }
    bool ok = layout->primitive == PL_PRIM_TRIANGLE_STRIP && !layout->indexed &&
        layout->vertex_count == 4 && layout->stride > 0 && layout->stride <= 1024 &&
        layout->byte_size == layout->stride * 4 && layout->byte_size <= 4096 &&
        layout->num_attributes > 0 && layout->num_attributes <= PL_OPENGL_YUV_COMPARE_QUAD_MAX_ATTRIBS &&
        layout->position_attrib >= 0 && layout->position_attrib < layout->num_attributes;
    for (int n = 0; ok && n < 4; n++) {
        int original = c->params.quad_order_reverse ? 3 - n : n;
        ok = !memcmp(candidate + n * layout->stride,
                     baseline + original * layout->stride, layout->stride);
    }
    if (ok) {
        c->quad_baseline_data = pl_alloc(c, layout->byte_size);
        c->quad_candidate_data = pl_alloc(c, layout->byte_size);
        ok = c->quad_baseline_data && c->quad_candidate_data;
        if (ok) {
            memcpy(c->quad_baseline_data, baseline, layout->byte_size);
            memcpy(c->quad_candidate_data, candidate, layout->byte_size);
            c->quad_layout = *layout;
            c->quad_recorded = true;
        }
    }
    if (!ok)
        c->invalidated = true;
    RELEASE_CURRENT();
    if (p->error_epoch != before) {
        c->invalidated = true;
        ok = false;
    }
    return ok;
}

bool pl_opengl_yuv_compare_quad_payloads(pl_gpu gpu, pl_tex target,
    const uint8_t **baseline, const uint8_t **candidate,
    struct pl_opengl_yuv_compare_quad_layout *layout)
{
    if (!gpu || !target || !pl_opengl_get(gpu) || !baseline || !candidate || !layout)
        return false;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.quad_order || !c->result.quad_order || !c->quad_recorded ||
        !c->finished || !c->last_baseline_ok || !c->quad_baseline_data || !c->quad_candidate_data)
        return false;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return false;
    }
    bool ok = yuv_compare_owner(c) && yuv_compare_epoch(gpu, c, "yuv_compare_quad_payloads");
    if (ok) {
        *baseline = c->quad_baseline_data;
        *candidate = c->quad_candidate_data;
        *layout = c->quad_layout;
    }
    RELEASE_CURRENT();
    return ok;
}

// CPU-only receipt data. Even a context failure must leave the failed draw
// recorded without issuing any further GL command in the foreign context.
static void yuv_compare_record(pl_gpu gpu, pl_tex target, unsigned int draw,
    pl_pass program, bool ok, uint64_t before, uint64_t after)
{
    (void) gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || c->finished || draw >= c->result.repetitions * 2 || draw >= 4)
        return;
    c->result.draw_ok[draw] = ok && !c->invalidated;
    c->result.program_tokens[draw] = (uintptr_t) program;
    c->result.draw_epoch_before[draw] = before;
    c->result.draw_epoch_after[draw] = after;
}

static void yuv_compare_finish(pl_gpu gpu, pl_tex target, bool b_ok, bool a_ok,
                                uint64_t body_hash)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->consumed || c->finished)
        return;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return;
    }
    if (!yuv_compare_owner(c) || c->invalidated) {
        RELEASE_CURRENT();
        return;
    }
    struct pl_gl *p = PL_PRIV(gpu);
    const gl_funcs *gl = gl_funcs_get(gpu);
    c->result.candidate_ok = b_ok && !c->invalidated;
    c->result.baseline_ok = a_ok && !c->invalidated;
    unsigned int last = c->params.uv_oracle
        ? c->result.repetitions * 2 - 2 : c->result.repetitions * 2 - 1;
    c->last_baseline_ok = last < 4 && c->result.draw_ok[last] &&
        c->result.program_tokens[last] && c->result.draw_epoch_after[last] == p->error_epoch &&
        !c->invalidated;
    if (c->params.sample_tap && (!c->result.sample_tap || c->result.sample_sites != 1 ||
        c->result.words_per_pixel != 26 || !c->original_body || !c->tap_body))
    {
        c->result.candidate_ok = c->result.baseline_ok = false;
        c->last_baseline_ok = false;
    }
    if (c->params.invariant_b && (!c->result.invariant_b || !c->params.sample_tap ||
        !c->candidate_vertex_source || !c->vertex_source || !c->result.invariant_uv_identifier[0]))
    {
        c->result.candidate_ok = c->result.baseline_ok = false;
        c->last_baseline_ok = false;
    }
    if (c->params.quad_order != c->result.quad_order ||
        c->params.quad_order_reverse != c->result.quad_order_reverse ||
        (c->params.quad_order && (!c->quad_recorded || !c->quad_baseline_data || !c->quad_candidate_data)))
    {
        c->result.candidate_ok = c->result.baseline_ok = false;
        c->last_baseline_ok = false;
    }
    if (c->params.uv_oracle_single_pair != c->result.uv_oracle_single_pair ||
        (c->params.uv_oracle &&
         (c->params.uv_oracle_single_pair ? c->result.repetitions != 1 : c->result.repetitions != 2)))
    {
        c->result.candidate_ok = c->result.baseline_ok = false;
        c->last_baseline_ok = false;
    }
    if (c->params.uv_oracle) {
        bool barriers = c->result.uv_oracle && c->result.repetitions >= 1 &&
            c->result.repetitions <= 2 && c->result.uv_oracle_barriers == c->result.repetitions &&
            c->result.uv_oracle_use_a == c->params.uv_oracle_use_a;
        for (unsigned int n = 0; n < c->result.repetitions && n < 2; n++) {
            barriers &= c->result.uv_oracle_barrier_ok[n] &&
                c->result.uv_oracle_barrier_epoch_before[n] == c->result.draw_epoch_after[n * 2] &&
                c->result.uv_oracle_barrier_epoch_after[n] == c->result.draw_epoch_before[n * 2 + 1];
        }
        c->result.candidate_ok &= barriers;
        c->result.baseline_ok &= barriers;
        c->last_baseline_ok &= barriers;
    }
    if (c->result.repetitions == 2 || c->params.uv_oracle) {
        bool clean_repeat = c->result.repetitions >= 1 && c->result.repetitions <= 2;
        for (unsigned int n = 0; n < c->result.repetitions * 2 && n < 4; n++) {
            clean_repeat &= c->result.draw_ok[n] && c->result.program_tokens[n] &&
                c->result.draw_epoch_before[n] == c->result.draw_epoch_after[n];
            if (n)
                clean_repeat &= c->result.draw_epoch_before[n] == c->result.draw_epoch_after[n-1];
        }
        if (c->result.repetitions == 2)
            clean_repeat &= c->result.program_tokens[0] == c->result.program_tokens[2] &&
                            c->result.program_tokens[1] == c->result.program_tokens[3];
        c->result.candidate_ok &= clean_repeat;
        c->result.baseline_ok &= clean_repeat;
        // Oracle and repeated experiments are all-or-nothing. Only original CSC
        // may cover B; no failed draw can be hidden by granting a fresh receipt.
        c->last_baseline_ok &= c->result.candidate_ok && c->result.baseline_ok;
    }
    c->result.shader_body_hash = body_hash;
    uint64_t before = p->error_epoch;
    gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    c->fence = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl->Flush(); // diagnostic completion only, never on the default path
    bool clean = gl_check_err(gpu, "yuv_compare_finish");
    if (!c->fence || !clean || p->error_epoch != before) {
        c->result.candidate_ok = c->result.baseline_ok = false;
        c->last_baseline_ok = false;
    }
    c->finish_epoch = p->error_epoch;
    c->finished = true;
    RELEASE_CURRENT();
}

static void yuv_compare_sources(pl_gpu gpu, pl_tex target, const char *b,
                                const char *a, const char *vertex)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    if (!t->compare)
        return;
    if (!MAKE_CURRENT()) {
        t->compare->invalidated = true;
        return;
    }
    struct gl_yuv_compare *c = t->compare;
    bool owner = yuv_compare_owner(c);
    RELEASE_CURRENT();
    if (!owner)
        return;
    c->candidate_source = pl_strdup0(c, pl_str0(b));
    c->baseline_source = pl_strdup0(c, pl_str0(a));
    c->vertex_source = pl_strdup0(c, pl_str0(vertex));
}

bool pl_opengl_yuv_compare_sources(pl_gpu gpu, pl_tex target,
    const char **candidate, const char **baseline, const char **vertex)
{
    if (!gpu || !target || !pl_opengl_get(gpu) || !candidate || !baseline || !vertex)
        return false;
    struct pl_tex_gl *t = PL_PRIV(target);
    if (!t->compare || !t->compare->candidate_source ||
        !t->compare->baseline_source || !t->compare->vertex_source)
        return false;
    if (!MAKE_CURRENT()) {
        t->compare->invalidated = true;
        return false;
    }
    bool owner = yuv_compare_owner(t->compare);
    RELEASE_CURRENT();
    if (!owner)
        return false;
    *candidate = t->compare->candidate_source;
    *baseline = t->compare->baseline_source;
    *vertex = t->compare->vertex_source;
    return true;
}

struct pl_yuv_compare *gl_tex_yuv_compare(pl_gpu gpu, pl_tex target)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    if (!t->compare)
        return NULL;
    if (!MAKE_CURRENT()) {
        t->compare->invalidated = true;
        t->compare->dispatch.rejected = true;
        return &t->compare->dispatch;
    }
    bool owner = yuv_compare_owner(t->compare);
    RELEASE_CURRENT();
    if (!owner || t->compare->invalidated) {
        t->compare->dispatch.rejected = true;
        return &t->compare->dispatch;
    }
    if (t->compare->consumed) {
        t->compare->invalidated = true;
        t->compare->result.candidate_ok = t->compare->result.baseline_ok = false;
        t->compare->dispatch.rejected = true;
        return &t->compare->dispatch;
    }
    t->compare->consumed = true;
    return &t->compare->dispatch;
}

// Private capture allocation. This is deliberately not a new ordinary GPU
// capability: the generic limits remain unchanged for renderer planning. The
// fixed params have no imports/exports, uniform/drawable/texel buffers, persistent
// mapping, or shared-memory offsets. Their only applicable size constraints are
// positive generic buffer/SSBO size, both bounded by the measured core SSBO max.
// The usual OpenGL allocator, error epoch and destroy lifecycle still apply.
static pl_buf yuv_compare_storage(pl_gpu gpu, struct gl_yuv_compare *c,
                                   size_t size, const void *zero)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!p->yuv_compare_caps_ready || !c || !zero || !size ||
        c->result.words_per_pixel != (c->params.sample_tap ? 26u : 14u) ||
        !c->storage_words || c->storage_words > (size_t) 256 * 256 * c->result.words_per_pixel ||
        size != c->storage_words * sizeof(uint32_t) ||
        size > p->yuv_compare_max_ssbo || size > PTRDIFF_MAX || !yuv_compare_owner(c) ||
        gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0))
        return NULL;
    return gl_buf_create(gpu, pl_buf_params(
        .size = size, .storable = true, .initial_data = zero));
}

// Fixed B-only sidecar, bounded independently of original A0 storage layout.
static pl_buf yuv_compare_probe_storage(pl_gpu gpu, struct gl_yuv_compare *c,
                                         const void *zero)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!c || !c->params.uv_oracle_probe || !c->params.uv_oracle ||
        !c->params.repeat || !c->params.sample_tap || !zero ||
        !p->yuv_compare_caps_ready || c->result.words_per_pixel != 26 ||
        !c->result.uv_oracle_probe_words ||
        c->result.repetitions < 1 || c->result.repetitions > 2 ||
        (c->params.uv_oracle_single_pair ? c->result.repetitions != 1 : c->result.repetitions != 2) ||
        c->result.uv_oracle_probe_words != c->storage_words / 26 * 10 * c->result.repetitions ||
        c->result.uv_oracle_probe_words > (size_t) 256 * 256 * 10 * 2 ||
        c->result.uv_oracle_probe_words > PTRDIFF_MAX / sizeof(uint32_t) ||
        c->result.uv_oracle_probe_words * sizeof(uint32_t) > p->yuv_compare_max_ssbo ||
        !yuv_compare_owner(c) || gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0))
        return NULL;
    return gl_buf_create(gpu, pl_buf_params(
        .size = c->result.uv_oracle_probe_words * sizeof(uint32_t),
        .storable = true, .initial_data = zero));
}

// Explicit arm only: log CPU metadata without any additional GL queries.
static bool yuv_compare_arm_reject(pl_gpu gpu, pl_tex target,
    const struct pl_opengl_yuv_compare_params *q, const char *reason)
{
    struct pl_gl *p = PL_PRIV(gpu);
    const struct pl_tex_gl *t = PL_PRIV(target);
    const struct pl_tex_gl *a_gl = PL_PRIV(q->baseline);
    const struct pl_tex_params *a = &q->baseline->params;
    pl_fmt f = a->format;
    PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-rejected reason=%s id=%"PRIu64
           " generation=%"PRIu64" gles=%d target_fbo=%u baseline_fbo=%u"
           " target=%dx%d baseline=%dx%dx%d renderable=%d target_yuv=%d"
           " existing_capture=%d production_draws=%"PRIu64" roi=%d,%d,%d,%d",
           reason, q->capture_id, q->generation, p->gles_ver, t->fbo, a_gl->fbo,
           target->params.w, target->params.h, a->w, a->h, a->d, a->renderable,
           t->rgb16f_to_yuv, !!t->compare, t->yuv_successful_draws,
           q->x, q->y, q->width, q->height);
    PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-format format=%s type=%d"
           " components=%d depths=%d,%d,%d,%d max_ssbo=%zu max_buffer=%zu"
           " private_ssbo=%zu requested_bytes=%zu error_epoch=%"PRIu64,
           f && f->name ? f->name : "null", f ? (int) f->type : -1,
           f ? f->num_components : 0, f ? f->component_depth[0] : 0,
           f ? f->component_depth[1] : 0, f ? f->component_depth[2] : 0,
           f ? f->component_depth[3] : 0, gpu->limits.max_ssbo_size,
           gpu->limits.max_buf_size, p->yuv_compare_max_ssbo,
           q->width > 0 && q->height > 0 && q->width <= 256 && q->height <= 256
               ? (size_t) q->width * q->height * (q->sample_tap ? 26 : 14) * sizeof(uint32_t) : 0,
           p->error_epoch);
    return false;
}

bool pl_opengl_yuv_compare_arm(pl_gpu gpu, pl_tex target,
                              const struct pl_opengl_yuv_compare_params *q)
{
    if (!gpu || !target || !q || !q->baseline) {
        if (gpu)
            PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-rejected reason=missing-argument");
        return false;
    }
    if (!pl_opengl_get(gpu)) {
        PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-rejected reason=not-opengl");
        return false;
    }
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *t = PL_PRIV(target);
    const struct pl_tex_params *a = &q->baseline->params;
    const struct pl_tex_gl *a_gl = PL_PRIV(q->baseline);
    const char *reason = NULL;
    if (q->baseline == target) reason = "same-texture";
    else if (!a_gl->fbo) reason = "baseline-fbo-zero";
    else if (a_gl->fbo == t->fbo) reason = "same-fbo";
    else if (!t->rgb16f_to_yuv) reason = "target-not-yuv";
    else if (t->compare) reason = "already-armed";
    else if (t->yuv_successful_draws) reason = "target-not-fresh";
    else if (p->gles_ver < 31) reason = "gles-below-31";
    else if (!p->yuv_compare_caps_ready) reason = "diagnostic-caps-not-ready";
    else if (q->quad_order && (!q->uv_oracle_single_pair || q->uv_oracle_use_a))
        reason = "quad-order-requires-single-own-uv";
    else if (q->quad_order_reverse && !q->quad_order) reason = "quad-reverse-without-framework";
    else if (q->uv_oracle_single_pair && (!q->uv_oracle_probe || !q->uv_oracle ||
             !q->sample_tap || !q->repeat)) reason = "oracle-single-requires-probe-oracle-repeat-tap";
    else if (q->uv_oracle_probe && !q->uv_oracle) reason = "oracle-probe-requires-oracle";
    else if (q->uv_oracle && (!q->sample_tap || !q->repeat || q->invariant_b))
        reason = "uv-oracle-requires-repeat-tap-no-invariant";
    else if (q->uv_oracle_use_a && !q->uv_oracle) reason = "uv-oracle-value-without-mode";
    else if (q->invariant_b && (!q->sample_tap || !q->repeat))
        reason = "invariant-b-requires-repeat-tap";
    else if (q->sample_tap && !q->repeat) reason = "sample-tap-requires-repeat";
    else if (!q->capture_id || !q->generation) reason = "zero-identity";
    else if (q->width < 1 || q->width > 256 || q->height < 1 || q->height > 256 ||
             q->x < 0 || q->y < 0 || q->x > target->params.w - q->width ||
             q->y > target->params.h - q->height) reason = "roi-out-of-bounds";
    else if (a->w != target->params.w || a->h != target->params.h || a->d)
        reason = "baseline-extent";
    else if (!a->renderable) reason = "baseline-not-renderable";
    else if (!a->format || a->format->type != PL_FMT_FLOAT ||
             a->format->num_components != 4 || a->format->component_depth[0] != 16 ||
             a->format->component_depth[1] != 16 || a->format->component_depth[2] != 16 ||
             a->format->component_depth[3] != 16) reason = "baseline-not-rgba16f";
    else if (pl_tex_rgb16f_to_yuv(gpu, q->baseline)) reason = "baseline-is-yuv";
    if (reason)
        return yuv_compare_arm_reject(gpu, target, q, reason);
    if (!MAKE_CURRENT())
        return yuv_compare_arm_reject(gpu, target, q, "make-current-failed");
    EGLDisplay owner_display = eglGetCurrentDisplay ? eglGetCurrentDisplay() : EGL_NO_DISPLAY;
    EGLContext owner_context = eglGetCurrentContext ? eglGetCurrentContext() : EGL_NO_CONTEXT;
    EGLSurface owner_draw = eglGetCurrentSurface ? eglGetCurrentSurface(EGL_DRAW) : EGL_NO_SURFACE;
    EGLSurface owner_read = eglGetCurrentSurface ? eglGetCurrentSurface(EGL_READ) : EGL_NO_SURFACE;
    if (!p->egl_dpy || !p->egl_ctx || owner_display != p->egl_dpy ||
        owner_context != p->egl_ctx || owner_draw == EGL_NO_SURFACE ||
        owner_read == EGL_NO_SURFACE) {
        PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-owner configured_display=%p"
               " configured_context=%p current_display=%p current_context=%p"
               " current_draw=%p current_read=%p",
               (void *) p->egl_dpy, (void *) p->egl_ctx,
               (void *) owner_display, (void *) owner_context,
               (void *) owner_draw, (void *) owner_read);
        RELEASE_CURRENT();
        return yuv_compare_arm_reject(gpu, target, q, "egl-owner");
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    // This diagnostic uses transient MapBufferRange. Do not allocate through
    // the immutable ARB storage path, whose ordinary buffers lack MAP_READ.
    if (gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0)) {
        RELEASE_CURRENT();
        return yuv_compare_arm_reject(gpu, target, q, "immutable-storage-map-unsupported");
    }
    uint64_t before = p->error_epoch;
    GLint blocks = 0, bindings = 0;
    gl->GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &blocks);
    gl->GetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &bindings);
    bool clean = gl_check_err(gpu, "yuv_compare_limits");
    int required = q->uv_oracle_probe ? 2 : 1;
    if (blocks < required || bindings < required || !gl->MapBufferRange || !gl->FenceSync ||
        !clean || p->error_epoch != before) {
        PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-limits fragment_blocks=%d"
               " bindings=%d map_proc=%d fence_proc=%d clean=%d"
               " epoch_before=%"PRIu64" epoch_after=%"PRIu64,
               blocks, bindings, !!gl->MapBufferRange, !!gl->FenceSync,
               clean, before, p->error_epoch);
        reason = blocks < required || bindings < required ? "ssbo-limits" :
                 !gl->MapBufferRange || !gl->FenceSync ? "missing-readback-proc" :
                 !clean ? "limits-gl-error" : "limits-epoch-change";
        RELEASE_CURRENT();
        return yuv_compare_arm_reject(gpu, target, q, reason);
    }
    if (q->sample_tap) {
        GLint range[2] = {0}, precision = 0;
        if (!gl->GetShaderPrecisionFormat) {
            RELEASE_CURRENT();
            return yuv_compare_arm_reject(gpu, target, q, "sample-tap-missing-precision-query");
        }
        gl->GetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &precision);
        clean = gl_check_err(gpu, "yuv_compare_tap_precision");
        if (!clean || p->error_epoch != before || precision < 23 || range[0] < 126 || range[1] < 127) {
            PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=sample-tap-rejected reason=highp-default"
                   " precision=%d range=%d,%d clean=%d", precision, range[0], range[1], clean);
            RELEASE_CURRENT();
            return yuv_compare_arm_reject(gpu, target, q, "sample-tap-highp-unsupported");
        }
    }
    struct gl_yuv_compare *c = pl_zalloc_ptr(NULL, c);
    c->params = *q;
    c->owner_display = owner_display;
    c->owner_context = owner_context;
    c->owner_draw = owner_draw;
    c->owner_read = owner_read;
    unsigned int words_per_pixel = q->sample_tap ? 26 : 14;
    c->storage_words = (size_t) q->width * q->height * words_per_pixel;
    unsigned int repetitions = q->uv_oracle_single_pair ? 1 : q->repeat ? 2 : 1;
    c->result = (struct pl_opengl_yuv_compare_result) {
        .capture_id = q->capture_id, .generation = q->generation,
        .words = c->storage_words * repetitions,
        .repetitions = repetitions,
        .sample_tap = q->sample_tap, .words_per_pixel = words_per_pixel,
        .invariant_b = q->invariant_b,
        .uv_oracle = q->uv_oracle, .uv_oracle_use_a = q->uv_oracle_use_a,
        .uv_oracle_probe = q->uv_oracle_probe,
        .uv_oracle_single_pair = q->uv_oracle_single_pair,
        .quad_order = q->quad_order, .quad_order_reverse = q->quad_order_reverse,
        .uv_oracle_probe_words = q->uv_oracle_probe
            ? (size_t) q->width * q->height * 10 * repetitions : 0,
    };
    size_t size = c->storage_words * sizeof(uint32_t);
    void *zero = pl_zalloc(NULL, size);
    GLint old_array = 0;
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &old_array);
    c->dispatch = (struct pl_yuv_compare) {
        .baseline = q->baseline, .x = q->x, .y = q->y,
        .width = q->width, .height = q->height,
        .checkpoint = pl_opengl_error_epoch, .finish = yuv_compare_finish,
        .record = yuv_compare_record, .repetitions = c->result.repetitions,
        .bodies = yuv_compare_bodies, .sample_tap = q->sample_tap,
        .vertices = yuv_compare_vertices, .invariant_b = q->invariant_b,
        .uv_oracle = q->uv_oracle, .uv_oracle_use_a = q->uv_oracle_use_a,
        .uv_oracle_probe = q->uv_oracle_probe,
        .uv_oracle_single_pair = q->uv_oracle_single_pair,
        .quad_order = q->quad_order, .quad_order_reverse = q->quad_order_reverse,
        .quad_payloads = yuv_compare_quad_payloads,
        .barrier = yuv_compare_barrier,
        .words_per_pixel = words_per_pixel,
        .sources = yuv_compare_sources, .prepare = yuv_compare_prepare,
        .storage = yuv_compare_storage(gpu, c, size, zero),
    };
    if (repetitions == 2 && c->dispatch.storage)
        c->dispatch.storage_repeat = yuv_compare_storage(gpu, c, size, zero);
    if (q->uv_oracle_probe && c->dispatch.storage &&
        (repetitions == 1 || c->dispatch.storage_repeat))
        c->dispatch.probe_storage = yuv_compare_probe_storage(gpu, c, zero);
    gl->BindBuffer(GL_ARRAY_BUFFER, old_array);
    pl_free(zero);
    clean = gl_check_err(gpu, "yuv_compare_arm");
    if (!c->dispatch.storage || (repetitions == 2 && !c->dispatch.storage_repeat) ||
        (q->uv_oracle_probe && !c->dispatch.probe_storage) || !clean || p->error_epoch != before) {
        PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=arm-storage allocated=%d clean=%d"
               " epoch_before=%"PRIu64" epoch_after=%"PRIu64,
               !!c->dispatch.storage, clean, before, p->error_epoch);
        reason = !c->dispatch.storage ? "storage-allocation" :
                 repetitions == 2 && !c->dispatch.storage_repeat ? "repeat-storage-allocation" :
                 q->uv_oracle_probe && !c->dispatch.probe_storage ? "probe-storage-allocation" :
                 !clean ? "storage-gl-error" : "storage-epoch-change";
        pl_buf_destroy(gpu, &c->dispatch.storage);
        pl_buf_destroy(gpu, &c->dispatch.storage_repeat);
        pl_buf_destroy(gpu, &c->dispatch.probe_storage);
        pl_free(c);
        RELEASE_CURRENT();
        return yuv_compare_arm_reject(gpu, target, q, reason);
    }
    t->compare = c;
    RELEASE_CURRENT();
    return true;
}

bool pl_opengl_yuv_compare_baseline_ok(pl_gpu gpu, pl_tex target)
{
    if (!gpu || !target || !pl_opengl_get(gpu))
        return false;
    struct pl_tex_gl *t = PL_PRIV(target);
    if (!MAKE_CURRENT()) {
        if (t->compare)
            t->compare->invalidated = true;
        return false;
    }
    struct gl_yuv_compare *c = t->compare;
    bool ok = false;
    if (c && yuv_compare_owner(c)) {
        bool clean = yuv_compare_epoch(gpu, c, "yuv_compare_baseline_ok");
        ok = clean && c->finished && c->last_baseline_ok;
    }
    RELEASE_CURRENT();
    return ok;
}

// CPU-only failure diagnostics over already mapped/copy-owned words. Object
// names below are allocation metadata, not a GL binding-query witness. Completion
// still means readback completed; the caller's strict coverage gate rejects data.
static void yuv_compare_report_counts(pl_gpu gpu, const struct gl_yuv_compare *c,
                                      const uint32_t *words)
{
    if (!c->params.uv_oracle || !words || c->result.words_per_pixel != 26 ||
        (c->result.repetitions != 1 && c->result.repetitions != 2) || c->storage_words % 26 ||
        c->result.words != c->storage_words * c->result.repetitions || !c->params.width ||
        !c->dispatch.storage ||
        (c->result.repetitions == 2 && !c->dispatch.storage_repeat))
        return;
    const unsigned int offsets[4] = {0, 10, 19, 25};
    unsigned int counts[2][4][3] = {{{0}}}; // zero, one, other; not a finite check
    size_t first_bad = SIZE_MAX;
    for (unsigned int n = 0; n < c->result.repetitions; n++) {
        for (size_t i = 0; i < c->storage_words; i += 26) {
            size_t absolute = n * c->storage_words + i;
            for (unsigned int k = 0; k < 4; k++) {
                uint32_t v = words[absolute + offsets[k]];
                counts[n][k][v < 2 ? v : 2]++;
                if (v != 1 && first_bad == SIZE_MAX)
                    first_bad = absolute;
            }
        }
    }
    if (first_bad == SIZE_MAX)
        return; // no new log or behavior for a valid capture
    for (unsigned int n = 0; n < c->result.repetitions; n++) {
        pl_buf storage = n ? c->dispatch.storage_repeat : c->dispatch.storage;
        const struct pl_buf_gl *buf = PL_PRIV(storage);
        PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=readback-counts chunk=%u"
            " buffer_object=%u buffer_offset=%zu buffer_size=%zu"
            " b_frag_zero=%u b_frag_one=%u b_frag_other=%u"
            " a_frag_zero=%u a_frag_one=%u a_frag_other=%u"
            " b_site_zero=%u b_site_one=%u b_site_other=%u"
            " a_site_zero=%u a_site_one=%u a_site_other=%u",
            n, buf->buffer, buf->offset, storage->params.size,
            counts[n][0][0], counts[n][0][1], counts[n][0][2],
            counts[n][1][0], counts[n][1][1], counts[n][1][2],
            counts[n][2][0], counts[n][2][1], counts[n][2][2],
            counts[n][3][0], counts[n][3][1], counts[n][3][2]);
    }
    size_t chunk = first_bad / c->storage_words;
    size_t pixel = (first_bad % c->storage_words) / 26;
    PL_ERR(gpu, "MKS-YUV-COMPARE-LIB event=readback-first-invalid chunk=%zu"
        " pixel=%zu roi_x=%zu roi_y=%zu absolute_word=%zu"
        " b_frag=%"PRIu32" a_frag=%"PRIu32" b_site=%"PRIu32" a_site=%"PRIu32
        " b_u_bits=%08"PRIx32" b_v_bits=%08"PRIx32
        " a_u_bits=%08"PRIx32" a_v_bits=%08"PRIx32,
        chunk, pixel, pixel % c->params.width, pixel / c->params.width, first_bad,
        words[first_bad], words[first_bad+10], words[first_bad+19], words[first_bad+25],
        words[first_bad+14], words[first_bad+15], words[first_bad+20], words[first_bad+21]);
}

enum pl_opengl_yuv_compare_status pl_opengl_yuv_compare_poll(
    pl_gpu gpu, pl_tex target, uint32_t *words, size_t capacity,
    struct pl_opengl_yuv_compare_result *result)
{
    if (!gpu || !target || !pl_opengl_get(gpu) || !result)
        return PL_OPENGL_YUV_COMPARE_FAILED;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c)
        return PL_OPENGL_YUV_COMPARE_FAILED;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (!yuv_compare_owner(c)) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    bool clean = yuv_compare_epoch(gpu, c, "yuv_compare_poll_begin");
    *result = c->result;
    if (!clean) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (!c->finished) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_PENDING;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (c->invalidated || !c->fence || !c->result.candidate_ok || !c->result.baseline_ok ||
        !words || capacity < c->result.words) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    GLenum ready = gl->ClientWaitSync(c->fence, 0, 0);
    if (ready == GL_TIMEOUT_EXPIRED) {
        clean = yuv_compare_epoch(gpu, c, "yuv_compare_poll_wait");
        RELEASE_CURRENT();
        return clean ? PL_OPENGL_YUV_COMPARE_PENDING : PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (ready != GL_ALREADY_SIGNALED && ready != GL_CONDITION_SATISFIED) {
        c->invalidated = true;
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    GLint old = 0;
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &old);
    bool ok = c->result.repetitions >= 1 && c->result.repetitions <= 2;
    for (unsigned int n = 0; ok && n < c->result.repetitions; n++) {
        pl_buf storage = n ? c->dispatch.storage_repeat : c->dispatch.storage;
        if (!storage) {
            ok = false;
            break;
        }
        struct pl_buf_gl *buf = PL_PRIV(storage);
        gl->BindBuffer(GL_ARRAY_BUFFER, buf->buffer);
        void *data = gl->MapBufferRange(GL_ARRAY_BUFFER, buf->offset,
            c->storage_words * sizeof(uint32_t), GL_MAP_READ_BIT);
        ok = data != NULL;
        if (data) {
            memcpy(words + n * c->storage_words, data,
                   c->storage_words * sizeof(uint32_t));
            ok = gl->UnmapBuffer(GL_ARRAY_BUFFER) == GL_TRUE;
        }
    }
    gl->BindBuffer(GL_ARRAY_BUFFER, old);
    ok = yuv_compare_epoch(gpu, c, "yuv_compare_poll") && ok;
    if (!ok)
        c->invalidated = true;
    if (ok)
        yuv_compare_report_counts(gpu, c, words);
    RELEASE_CURRENT();
    return ok ? PL_OPENGL_YUV_COMPARE_COMPLETE : PL_OPENGL_YUV_COMPARE_FAILED;
}

enum pl_opengl_yuv_compare_status pl_opengl_yuv_compare_probe_poll(
    pl_gpu gpu, pl_tex target, uint32_t *words, size_t capacity)
{
    if (!gpu || !target || !pl_opengl_get(gpu))
        return PL_OPENGL_YUV_COMPARE_FAILED;
    struct pl_tex_gl *t = PL_PRIV(target);
    struct gl_yuv_compare *c = t->compare;
    if (!c || !c->params.uv_oracle_probe || !c->result.uv_oracle_probe ||
        !c->dispatch.probe_storage || !words ||
        !c->result.uv_oracle_probe_words || capacity < c->result.uv_oracle_probe_words ||
        c->dispatch.probe_storage->params.size != c->result.uv_oracle_probe_words * sizeof(uint32_t))
        return PL_OPENGL_YUV_COMPARE_FAILED;
    if (!MAKE_CURRENT()) {
        c->invalidated = true;
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (!yuv_compare_owner(c)) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    bool clean = yuv_compare_epoch(gpu, c, "yuv_compare_probe_poll_begin");
    if (!clean || (c->finished && (!c->fence || !c->result.candidate_ok ||
                                  !c->result.baseline_ok))) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (!c->finished) {
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_PENDING;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    GLenum ready = gl->ClientWaitSync(c->fence, 0, 0);
    if (ready == GL_TIMEOUT_EXPIRED) {
        clean = yuv_compare_epoch(gpu, c, "yuv_compare_probe_poll_wait");
        RELEASE_CURRENT();
        return clean ? PL_OPENGL_YUV_COMPARE_PENDING : PL_OPENGL_YUV_COMPARE_FAILED;
    }
    if (ready != GL_ALREADY_SIGNALED && ready != GL_CONDITION_SATISFIED) {
        c->invalidated = true;
        RELEASE_CURRENT();
        return PL_OPENGL_YUV_COMPARE_FAILED;
    }
    struct pl_buf_gl *buf = PL_PRIV(c->dispatch.probe_storage);
    GLint old = 0;
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &old);
    gl->BindBuffer(GL_ARRAY_BUFFER, buf->buffer);
    void *data = gl->MapBufferRange(GL_ARRAY_BUFFER, buf->offset,
        c->result.uv_oracle_probe_words * sizeof(uint32_t), GL_MAP_READ_BIT);
    bool ok = data != NULL;
    if (data) {
        memcpy(words, data, c->result.uv_oracle_probe_words * sizeof(uint32_t));
        ok = gl->UnmapBuffer(GL_ARRAY_BUFFER) == GL_TRUE;
    }
    gl->BindBuffer(GL_ARRAY_BUFFER, old);
    ok = yuv_compare_epoch(gpu, c, "yuv_compare_probe_poll") && ok;
    if (!ok)
        c->invalidated = true;
    RELEASE_CURRENT();
    return ok ? PL_OPENGL_YUV_COMPARE_COMPLETE : PL_OPENGL_YUV_COMPARE_FAILED;
}

bool pl_opengl_error_epoch(pl_gpu gpu, uint64_t *out_epoch)
{
    if (!gpu || !out_epoch || !pl_opengl_get(gpu) || !MAKE_CURRENT())
        return false;
    struct pl_gl *p = PL_PRIV(gpu);
    gl_check_err(gpu, "pl_opengl_error_epoch");
    *out_epoch = p->error_epoch;
    RELEASE_CURRENT();
    return true;
}

bool pl_opengl_yuv_output_ok(pl_gpu gpu, pl_tex target)
{
    if (!gpu || !pl_opengl_get(gpu) || !target || !MAKE_CURRENT())
        return false;
    struct pl_gl *p = PL_PRIV(gpu);
    const struct pl_tex_gl *t = PL_PRIV(target);
    bool ok = gl_check_err(gpu, "pl_opengl_yuv_output_ok") &&
              t->rgb16f_to_yuv && t->yuv_successful_draws == 1 && !p->failed;
    RELEASE_CURRENT();
    return ok;
}

unsigned int pl_opengl_unwrap(pl_gpu gpu, pl_tex tex,
                              unsigned int *out_target, int *out_iformat,
                              unsigned int *out_fbo)
{
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (!tex_gl->texture) {
        PL_ERR(gpu, "Trying to call `pl_opengl_unwrap` on a pseudo-texture "
               "(perhaps obtained by `pl_swapchain_start_frame`?)");
        return 0;
    }

    if (out_target)
        *out_target = tex_gl->target;
    if (out_iformat)
        *out_iformat = tex_gl->iformat;
    if (out_fbo)
        *out_fbo = tex_gl->fbo;

    return tex_gl->texture;
}

void gl_tex_invalidate(pl_gpu gpu, pl_tex tex)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    if (!MAKE_CURRENT())
        return;

    if (tex_gl->texture && p->has_invalidate_tex)
        gl->InvalidateTexImage(tex_gl->texture, 0);

    if ((tex_gl->wrapped_fb || tex_gl->fbo) && p->has_invalidate_fb) {
        GLenum attachment = tex_gl->fbo ? GL_COLOR_ATTACHMENT0 : GL_COLOR;
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, tex_gl->fbo);
        gl->InvalidateFramebuffer(GL_DRAW_FRAMEBUFFER, 1, &attachment);
        gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    gl_check_err(gpu, "gl_tex_invalidate");
    RELEASE_CURRENT();
}

void gl_tex_clear_ex(pl_gpu gpu, pl_tex tex, const union pl_clear_color color)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    pl_assert(tex_gl->fbo || tex_gl->wrapped_fb);
    if (tex_gl->rgb16f_to_yuv) {
        PL_ERR(gpu, "YUV drawable does not support clears");
        p->failed = true;
        p->error_epoch++;
        RELEASE_CURRENT();
        return;
    }

    const GLenum target = p->gles_ver && p->gles_ver < 30 ?
        GL_FRAMEBUFFER : GL_DRAW_FRAMEBUFFER;

    gl->BindFramebuffer(target, tex_gl->fbo);

    switch (tex->params.format->type) {
    case PL_FMT_UNKNOWN:
    case PL_FMT_FLOAT:
    case PL_FMT_UNORM:
    case PL_FMT_SNORM:
        gl->ClearColor(color.f[0], color.f[1], color.f[2], color.f[3]);
        gl->Clear(GL_COLOR_BUFFER_BIT);
        break;

    case PL_FMT_UINT: {
        GLuint gl_color[] = { color.u[0], color.u[1], color.u[2], color.u[3] };
        gl->ClearBufferuiv(GL_COLOR, 0, gl_color);
        break;
    }

    case PL_FMT_SINT: {
        GLint gl_color[] = { color.i[0], color.i[1], color.i[2], color.i[3] };
        gl->ClearBufferiv(GL_COLOR, 0, gl_color);
        break;
    }

    case PL_FMT_TYPE_COUNT:
        pl_unreachable();
    }

    gl->BindFramebuffer(target, 0);
    gl_check_err(gpu, "gl_tex_clear");
    RELEASE_CURRENT();
}

void gl_tex_blit(pl_gpu gpu, const struct pl_tex_blit_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!MAKE_CURRENT())
        return;

    struct pl_tex_gl *src_gl = PL_PRIV(params->src);
    struct pl_tex_gl *dst_gl = PL_PRIV(params->dst);

    pl_assert(src_gl->fbo || src_gl->wrapped_fb);
    pl_assert(dst_gl->fbo || dst_gl->wrapped_fb);
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, src_gl->fbo);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, dst_gl->fbo);

    static const GLint filters[PL_TEX_SAMPLE_MODE_COUNT] = {
        [PL_TEX_SAMPLE_NEAREST] = GL_NEAREST,
        [PL_TEX_SAMPLE_LINEAR]  = GL_LINEAR,
    };

    pl_rect3d src_rc = params->src_rc, dst_rc = params->dst_rc;
    gl->BlitFramebuffer(src_rc.x0, src_rc.y0, src_rc.x1, src_rc.y1,
                        dst_rc.x0, dst_rc.y0, dst_rc.x1, dst_rc.y1,
                        GL_COLOR_BUFFER_BIT, filters[params->sample_mode]);

    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    gl_check_err(gpu, "gl_tex_blit");
    RELEASE_CURRENT();
}

static int get_alignment(size_t pitch)
{
    if (pitch % 8 == 0)
        return 8;
    if (pitch % 4 == 0)
        return 4;
    if (pitch % 2 == 0)
        return 2;
    return 1;
}

bool gl_tex_upload(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    pl_buf buf = params->buf;
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? PL_PRIV(buf) : NULL;

    // If the user requests asynchronous uploads, it's more efficient to do
    // them via a PBO - this allows us to skip blocking the caller, especially
    // when the host pointer can be imported directly.
    if (params->callback && !buf) {
        size_t buf_size = pl_tex_transfer_size(params);
        const size_t min_size = 32*1024; // 32 KiB
        if (buf_size >= min_size && buf_size <= gpu->limits.max_buf_size)
            return pl_tex_upload_pbo(gpu, params);
    }

    if (!MAKE_CURRENT())
        return false;

    uintptr_t src = (uintptr_t) params->ptr;
    if (buf) {
        gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_gl->buffer);
        src = buf_gl->offset + params->buf_offset;
    }

    bool misaligned = params->row_pitch % fmt->texel_size;
    int stride_w = params->row_pitch / fmt->texel_size;
    int stride_h = params->depth_pitch / params->row_pitch;

    int dims = pl_tex_params_dimension(tex->params);
    if (dims > 1)
        gl->PixelStorei(GL_UNPACK_ALIGNMENT, get_alignment(params->row_pitch));

    int rows = pl_rect_h(params->rc);
    if (misaligned) {
        rows = 1;
    } else if (stride_w != pl_rect_w(params->rc)) {
        gl->PixelStorei(GL_UNPACK_ROW_LENGTH, stride_w);
    }

    int imgs = pl_rect_d(params->rc);
    if (stride_h != pl_rect_h(params->rc) || rows < stride_h)
        gl->PixelStorei(GL_UNPACK_IMAGE_HEIGHT, stride_h);

    gl->BindTexture(tex_gl->target, tex_gl->texture);
    gl_timer_begin(gpu, params->timer);

    switch (dims) {
    case 1:
        gl->TexSubImage1D(tex_gl->target, 0, params->rc.x0, pl_rect_w(params->rc),
                          tex_gl->format, tex_gl->type, (void *) src);
        break;
    case 2:
        for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
            gl->TexSubImage2D(tex_gl->target, 0, params->rc.x0, y,
                              pl_rect_w(params->rc), rows, tex_gl->format,
                              tex_gl->type, (void *) src);
            src += params->row_pitch * rows;
        }
        break;
    case 3:
        for (int z = params->rc.z0; z < params->rc.z1; z += imgs) {
            uintptr_t row_src = src;
            for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
                gl->TexSubImage3D(tex_gl->target, 0, params->rc.x0, y, z,
                                  pl_rect_w(params->rc), rows, imgs,
                                  tex_gl->format, tex_gl->type, (void *) row_src);
                row_src = (uintptr_t) row_src + params->row_pitch * rows;
            }
            src += params->depth_pitch * imgs;
        }
        break;
    }

    gl_timer_end(gpu, params->timer);
    gl->BindTexture(tex_gl->target, 0);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 4);
    gl->PixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    gl->PixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);

    if (buf) {
        gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        if (buf->params.host_mapped) {
            // Make sure the PBO is not reused until GL is done with it. If a
            // previous operation is pending, "update" it by creating a new
            // fence that will cover the previous operation as well.
            gl->DeleteSync(buf_gl->fence);
            buf_gl->fence = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
    }

    if (params->callback) {
        PL_ARRAY_APPEND(gpu, p->callbacks, (struct gl_cb) {
            .sync = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0),
            .callback = params->callback,
            .priv = params->priv,
        });
    }

    bool ok = gl_check_err(gpu, "gl_tex_upload");
    RELEASE_CURRENT();
    return ok;
}

bool gl_tex_download(pl_gpu gpu, const struct pl_tex_transfer_params *params)
{
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    pl_tex tex = params->tex;
    pl_fmt fmt = tex->params.format;
    pl_buf buf = params->buf;
    struct pl_tex_gl *tex_gl = PL_PRIV(tex);
    struct pl_buf_gl *buf_gl = buf ? PL_PRIV(buf) : NULL;
    bool ok = true;

    if (params->callback && !buf) {
        size_t buf_size = pl_tex_transfer_size(params);
        const size_t min_size = 32*1024; // 32 KiB
        if (buf_size >= min_size && buf_size <= gpu->limits.max_buf_size)
            return pl_tex_download_pbo(gpu, params);
    }

    if (!MAKE_CURRENT())
        return false;

    uintptr_t dst = (uintptr_t) params->ptr;
    if (buf) {
        gl->BindBuffer(GL_PIXEL_PACK_BUFFER, buf_gl->buffer);
        dst = buf_gl->offset + params->buf_offset;
    }

    pl_rect3d full = {
        0, 0, 0,
        tex->params.w,
        PL_DEF(tex->params.h, 1),
        PL_DEF(tex->params.d, 1),
    };

    bool misaligned = params->row_pitch % fmt->texel_size;
    int stride_w = params->row_pitch / fmt->texel_size;
    int stride_h = params->depth_pitch / params->row_pitch;

    int dims = pl_tex_params_dimension(tex->params);
    bool is_copy = pl_rect3d_eq(params->rc, full) &&
                   stride_w == tex->params.w &&
                   stride_h == PL_DEF(tex->params.h, 1) &&
                   !misaligned;

    gl_timer_begin(gpu, params->timer);

    if (tex_gl->fbo || tex_gl->wrapped_fb) {
        // We can use a more efficient path when we have an FBO available
        if (dims > 1)
            gl->PixelStorei(GL_PACK_ALIGNMENT, get_alignment(params->row_pitch));

        int rows = pl_rect_h(params->rc);
        if (misaligned) {
            rows = 1;
        } else if (stride_w != tex->params.w) {
            gl->PixelStorei(GL_PACK_ROW_LENGTH, stride_w);
        }

        // No 3D framebuffers
        pl_assert(pl_rect_d(params->rc) == 1);

        const GLenum target = p->gles_ver && p->gles_ver < 30 ?
            GL_FRAMEBUFFER : GL_READ_FRAMEBUFFER;

        gl->BindFramebuffer(target, tex_gl->fbo);
        for (int y = params->rc.y0; y < params->rc.y1; y += rows) {
            gl->ReadPixels(params->rc.x0, y, pl_rect_w(params->rc), rows,
                           tex_gl->format, tex_gl->type, (void *) dst);
            dst += params->row_pitch * rows;
        }
        gl->BindFramebuffer(target, 0);
        gl->PixelStorei(GL_PACK_ALIGNMENT, 4);
        gl->PixelStorei(GL_PACK_ROW_LENGTH, 0);
    } else if (is_copy) {
        // We're downloading the entire texture
        gl->BindTexture(tex_gl->target, tex_gl->texture);
        gl->GetTexImage(tex_gl->target, 0, tex_gl->format, tex_gl->type, (void *) dst);
        gl->BindTexture(tex_gl->target, 0);
    } else {
        PL_ERR(gpu, "Partial downloads of 3D textures not implemented!");
        ok = false;
    }

    gl_timer_end(gpu, params->timer);

    if (buf) {
        gl->BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        if (ok && buf->params.host_mapped) {
            gl->DeleteSync(buf_gl->fence);
            buf_gl->fence = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        }
    }

    if (params->callback) {
        PL_ARRAY_APPEND(gpu, p->callbacks, (struct gl_cb) {
            .sync = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0),
            .callback = params->callback,
            .priv = params->priv,
        });
    }

    ok &= gl_check_err(gpu, "gl_tex_download");
    RELEASE_CURRENT();
    return ok;
}

// ---- Fixed-frame full-screen UV atlas numerical anchor (default off) ----
static bool uv_atlas_owner(pl_opengl_uv_atlas_anchor c)
{
    bool ok = eglGetCurrentDisplay && eglGetCurrentContext && eglGetCurrentSurface &&
        eglGetCurrentDisplay() == c->owner_display && eglGetCurrentContext() == c->owner_context &&
        eglGetCurrentSurface(EGL_DRAW) == c->owner_draw && eglGetCurrentSurface(EGL_READ) == c->owner_read;
    if (!ok) {
        PL_ERR(c->gpu, "MKS-ANCHOR-DIAG owner mismatch dpy=%p/%p ctx=%p/%p draw=%p/%p read=%p/%p",
               (void *) eglGetCurrentDisplay ? (void *) eglGetCurrentDisplay() : NULL, (void *) c->owner_display,
               (void *) eglGetCurrentContext ? (void *) eglGetCurrentContext() : NULL, (void *) c->owner_context,
               eglGetCurrentSurface ? (void *) eglGetCurrentSurface(EGL_DRAW) : NULL, (void *) c->owner_draw,
               eglGetCurrentSurface ? (void *) eglGetCurrentSurface(EGL_READ) : NULL, (void *) c->owner_read);
        c->invalidated = true;
    }
    return ok;
}

// Success leaves the normal GL context hold active for the caller to release.
// A finished production seed is sealed: the transaction's own cleanliness was
// verified op-by-op at finish (draw epochs, framebuffer identity where the
// diagnostic mode requires it, and its own error checks). After finish, the
// seeding frame keeps rendering through its ordinary final pass and swap; a
// benign recoverable error left there can be consumed by unrelated code on
// the next frame before this anchor is polled, bumping the global error
// epoch. That drift is not the sealed transaction's failure, so the global
// epoch equality below is enforced only while the transaction is still open.
// Explicit failures (fence loss, owner/context loss) still invalidate.
static bool uv_atlas_sealed(pl_opengl_uv_atlas_anchor c)
{
    return c->production_seed && c->finished && c->baseline_ok;
}

static bool uv_atlas_enter(pl_opengl_uv_atlas_anchor c)
{
    pl_gpu gpu = c->gpu;
    if (!MAKE_CURRENT()) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG enter make-current");
        c->invalidated = true; return false;
    }
    if (!uv_atlas_owner(c) || c->invalidated) {
        if (!c->invalidated)
            PL_ERR(gpu, "MKS-ANCHOR-DIAG enter unexpected");
        RELEASE_CURRENT(); return false;
    }
    if (uv_atlas_sealed(c))
        return true;
    struct pl_gl *p = PL_PRIV(gpu);
    bool clean = gl_check_err(gpu, "uv_atlas_anchor");
    if (!clean || p->error_epoch != c->entry_epoch) {
        c->invalidated = true; RELEASE_CURRENT(); return false;
    }
    return true;
}

static bool uv_atlas_leave(pl_opengl_uv_atlas_anchor c)
{
    pl_gpu gpu=c->gpu;
    struct pl_gl *p=PL_PRIV(gpu);
    RELEASE_CURRENT();
    if (!uv_atlas_sealed(c) && p->error_epoch != c->entry_epoch) c->invalidated=true;
    return !c->invalidated;
}

// Pin the verified EGL owner across ordinary finalization, all shader/pass
// creation and all three submissions. Nested GL calls cannot invoke a new
// make_current callback until the transaction finishes, including error exits.
static void uv_atlas_drop_hold(pl_opengl_uv_atlas_anchor c)
{
    if (!c->transaction_hold) return;
    c->transaction_hold=false;
    if (!uv_atlas_leave(c)) c->baseline_ok=false;
}

static bool uv_atlas_checkpoint(pl_gpu gpu, pl_tex target, uint64_t *epoch)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || !epoch || !uv_atlas_enter(c)) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG checkpoint enter c=%d epoch_arg=%d",
               c ? 1 : 0, epoch ? 1 : 0);
        return false;
    }
    if (c->reader) {
        const gl_funcs *gl=gl_funcs_get(gpu);
        GLint generic=-1, slots[2]={-1,-1};
        gl->GetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING,&generic);
        for (int i=0;i<2;i++)
            gl->GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING,i,&slots[i]);
        bool clean=gl_check_err(gpu,"uv_atlas_read_slots");
        struct pl_gl *p=PL_PRIV(gpu);
        if (!clean || p->error_epoch!=c->entry_epoch || generic || slots[0] || slots[1]) {
            PL_ERR(gpu, "MKS-ANCHOR-DIAG checkpoint reader clean=%d epoch=%llu/%llu",
                   clean?1:0, (unsigned long long)p->error_epoch,
                   (unsigned long long)c->entry_epoch);
            c->invalidated=true; uv_atlas_leave(c); return false;
        }
    }
    struct pl_gl *p = PL_PRIV(gpu);
    *epoch = p->error_epoch;
    bool ok = !c->invalidated && p->error_epoch == c->entry_epoch;
    if (!ok)
        PL_ERR(gpu, "MKS-ANCHOR-DIAG checkpoint tail inv=%d epoch=%llu/%llu",
               c->invalidated?1:0, (unsigned long long)p->error_epoch,
               (unsigned long long)c->entry_epoch);
    RELEASE_CURRENT();
    return ok;
}

static void uv_atlas_record(pl_gpu gpu, pl_tex target, unsigned int stage, pl_pass program,
                            bool ok, uint64_t before, uint64_t after)
{
    (void) gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || c->finished || stage >= 3) return;
    c->result.draw_ok[stage] = ok && !c->invalidated;
    c->result.program_tokens[stage] = (uintptr_t) program;
    c->result.draw_epoch_before[stage] = before;
    c->result.draw_epoch_after[stage] = after;
    // Production seeds deliberately verify no framebuffer identity: the
    // pass implementation unbinds back to the default framebuffer when the
    // draw returns, so any post-draw ambient binding read here would be 0
    // and never match. Transaction integrity rides on the per-stage epoch
    // checkpoints and program tokens instead; do not "restore" a binding
    // check for production seeds without capturing it inside the pass run.
}

// Read every pixel from the same real RGB16F attachment. Save both framebuffer
// bindings and each affected pack/read state; target READ_BUFFER belongs to that
// FBO and must be restored before rebinding the old read FBO.
static bool uv_atlas_readback(pl_gpu gpu, pl_tex target, unsigned int stage)
{
    struct pl_tex_gl *t_pre = PL_PRIV(target);
    if (t_pre->atlas_anchor && t_pre->atlas_anchor->production_seed) {
        // Zero-readback production seed: the stage drew and its GL state was
        // verified by checkpoint/record; no CPU pixels are transferred.
        return true;
    }
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || stage >= 3 || !c->result.draw_ok[stage] || !uv_atlas_enter(c)) return false;
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    uint64_t before = p->error_epoch;
    c->result.readback_epoch_before[stage] = before;
    GLint old_read=0, old_draw=0, old_read_buffer=0, target_read_buffer=0;
    GLint pack_buffer=0, alignment=0, row_length=0, skip_rows=0, skip_pixels=0;
    gl->GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &old_read);
    gl->GetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_draw);
    gl->GetIntegerv(GL_READ_BUFFER, &old_read_buffer);
    gl->GetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pack_buffer);
    gl->GetIntegerv(GL_PACK_ALIGNMENT, &alignment);
    gl->GetIntegerv(GL_PACK_ROW_LENGTH, &row_length);
    gl->GetIntegerv(GL_PACK_SKIP_ROWS, &skip_rows);
    gl->GetIntegerv(GL_PACK_SKIP_PIXELS, &skip_pixels);
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, c->fbo);
    gl->GetIntegerv(GL_READ_BUFFER, &target_read_buffer);
    gl->ReadBuffer(GL_COLOR_ATTACHMENT0);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, c->fbo);
    GLint actual=0, samples=0, sample_buffers=0, type=0, object=0, object_type=0;
    gl->GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &actual);
    gl->GetIntegerv(GL_SAMPLES, &samples);
    gl->GetIntegerv(GL_SAMPLE_BUFFERS, &sample_buffers);
    gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &type);
    gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &object);
    gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &object_type);
    const GLenum sizes[] = {GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE,
                           GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE};
    bool ok = actual == (GLint)c->fbo && actual != 0 && samples == 0 && sample_buffers == 0 &&
        type == GL_FLOAT && object != 0 && (object_type == GL_TEXTURE || object_type == GL_RENDERBUFFER) &&
        gl->CheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    for (unsigned int n=0; n<4; n++) {
        GLint bits=0;
        gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, sizes[n], &bits);
        ok &= bits == 16;
    }
    if (object_type == GL_TEXTURE) {
        GLint level=-1, layer=-1, face=-1;
        gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &level);
        gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &layer);
        gl->GetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &face);
        ok &= level == 0 && layer == 0 && face == 0;
    }
    if (!stage) { c->attachment = object; c->attachment_type = object_type; }
    else ok &= c->attachment == (GLuint)object && c->attachment_type == (GLenum)object_type;
    c->result.framebuffer[stage] = actual;
    uint32_t *pixels = c->rgb + stage * c->result.rgba_words;
    for (size_t n=0; n<c->result.rgba_words; n++) pixels[n] = 0x7fc00000u;
    gl->BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    gl->PixelStorei(GL_PACK_ALIGNMENT, 1);
    gl->PixelStorei(GL_PACK_ROW_LENGTH, 0);
    gl->PixelStorei(GL_PACK_SKIP_ROWS, 0);
    gl->PixelStorei(GL_PACK_SKIP_PIXELS, 0);
    if (ok)
        gl->ReadPixels(0, 0, c->result.width, c->result.height, GL_RGBA, GL_FLOAT, pixels);
    if (!uv_atlas_owner(c)) { RELEASE_CURRENT(); return false; } // never restore in a foreign context
    gl->ReadBuffer(target_read_buffer);
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, old_read);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, old_draw);
    gl->BindBuffer(GL_PIXEL_PACK_BUFFER, pack_buffer);
    gl->PixelStorei(GL_PACK_ALIGNMENT, alignment);
    gl->PixelStorei(GL_PACK_ROW_LENGTH, row_length);
    gl->PixelStorei(GL_PACK_SKIP_ROWS, skip_rows);
    gl->PixelStorei(GL_PACK_SKIP_PIXELS, skip_pixels);
    GLint restored=0;
    gl->GetIntegerv(GL_READ_BUFFER, &restored);
    ok &= restored == old_read_buffer;
    bool clean = gl_check_err(gpu, "uv_atlas_readback");
    ok &= clean && p->error_epoch == before && before == c->entry_epoch;
    c->result.readback_epoch_after[stage] = p->error_epoch;
    c->result.readback_ok[stage] = ok;
    if (ok) {
        size_t bad=0;
        for (size_t n=0; n<c->result.rgba_words; n++) bad += !mks_atlas_half_exact(pixels[n]);
        c->result.bad_half_components[stage] = bad;
        c->result.half_exact[stage] = bad == 0;
        if (bad) {
            ok=false; c->result.readback_ok[stage]=false; c->invalidated=true;
            PL_ERR(gpu, "MKS-YUV-ATLAS-LIB event=rejected reason=readback-non-half stage=%u", stage);
        }
    } else {
        c->invalidated = true;
        PL_ERR(gpu, "MKS-YUV-ATLAS-LIB event=rejected reason=readback-state-or-format stage=%u", stage);
    }
    RELEASE_CURRENT();
    return ok && p->error_epoch == c->entry_epoch;
}

static void uv_atlas_finish(pl_gpu gpu, pl_tex target, bool ok)
{
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || c->finished) return;
    if (!uv_atlas_enter(c)) {
        c->finished=true; c->dispatch.ended=true; c->baseline_ok=false;
        uv_atlas_drop_hold(c);
        return;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    struct pl_gl *p = PL_PRIV(gpu);
    ok &= c->result.source_abi_ok && c->result.program_tokens[0] &&
        c->result.program_tokens[0] == c->result.program_tokens[2] &&
        c->result.program_tokens[0] != c->result.program_tokens[1] && c->result.program_tokens[1];
    for (unsigned int n=0; n<3; n++) {
        ok &= c->result.draw_ok[n] &&
            c->result.draw_epoch_before[n] == c->entry_epoch && c->result.draw_epoch_after[n] == c->entry_epoch;
        // Framebuffer identity is a diagnostic-mode contract only; see the
        // note in uv_atlas_record for why production never checks it.
        if (c->production_seed) continue;
        ok &= c->result.framebuffer[n] == c->fbo;
        ok &= c->result.readback_ok[n] && c->result.half_exact[n] &&
            !c->result.bad_half_components[n] &&
            c->result.readback_epoch_before[n] == c->entry_epoch && c->result.readback_epoch_after[n] == c->entry_epoch;
    }
    if (ok && !c->production_seed) {
        for (size_t n=0; n<c->result.rgba_words; n++) {
            c->result.mismatches_before_atlas += c->rgb[n] != c->rgb[c->result.rgba_words+n];
            c->result.mismatches_before_after += c->rgb[n] != c->rgb[c->result.rgba_words*2+n];
        }
    }
    if (ok) {
        // Production or not, completion is gated by the fence: the atlas
        // writes must be visible before any consumer samples the table.
        // Production seeds additionally read nothing back to the CPU.
        gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        c->fence = gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl->Flush();
        bool clean = gl_check_err(gpu, "uv_atlas_finish");
        ok &= c->fence && clean && p->error_epoch == c->entry_epoch;
        if (!ok)
            PL_ERR(gpu, "MKS-ANCHOR-DIAG finish fence=%d clean=%d epoch=%llu/%llu",
                   c->fence ? 1 : 0, clean ? 1 : 0,
                   (unsigned long long)p->error_epoch,
                   (unsigned long long)c->entry_epoch);
    } else if (c->production_seed) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG finish pre ok_in=0 abi=%d tok=%llu,%llu,%llu "
               "draw=%d,%d,%d depoch=%llu,%llu,%llu",
               c->result.source_abi_ok ? 1 : 0,
               (unsigned long long)c->result.program_tokens[0],
               (unsigned long long)c->result.program_tokens[1],
               (unsigned long long)c->result.program_tokens[2],
               c->result.draw_ok[0] ? 1 : 0, c->result.draw_ok[1] ? 1 : 0,
               c->result.draw_ok[2] ? 1 : 0,
               (unsigned long long)c->result.draw_epoch_before[1],
               (unsigned long long)c->result.draw_epoch_after[1],
               (unsigned long long)c->entry_epoch);
    }
    c->baseline_ok = ok;
    c->invalidated |= !ok;
    c->result.finish_epoch = p->error_epoch;
    c->finished = true;
    if (!uv_atlas_leave(c)) c->baseline_ok=false;
    c->dispatch.ended=true;
    uv_atlas_drop_hold(c);
}

static char *uv_atlas_strdup(const char *text)
{
    size_t bytes = strlen(text) + 1;
    char *copy = malloc(bytes);
    if (copy) memcpy(copy, text, bytes);
    return copy;
}

static bool uv_atlas_body(pl_gpu gpu, pl_tex target, const char *body)
{
    (void) gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || c->original_body || !body || !*body || strstr(body, "mks_") || strstr(body, "MKS_") ||
        !uv_atlas_enter(c)) return false;
    c->original_body = uv_atlas_strdup(body);
    c->dispatch.shader_body = c->original_body;
    bool ok = c->original_body != NULL;
    if (!ok) c->invalidated = true;
    return uv_atlas_leave(c) && ok;
}

static bool uv_atlas_begin(pl_gpu gpu, pl_tex target, const char *a0, const char *atlas,
    const char *vertex, const struct pl_opengl_yuv_compare_quad_layout *layout, const uint8_t *vertices)
{
    (void) gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || c->result.source_abi_ok || !a0 || !atlas || !vertex || !layout || !vertices ||
        !c->original_body || !strstr(a0, c->original_body) || !mks_atlas_clean_a0(a0) ||
        !strcmp(a0, atlas) || layout->vertex_count != 4 || layout->byte_size > 4096 ||
        !layout->byte_size || !uv_atlas_enter(c)) return false;
    c->a0_source = uv_atlas_strdup(a0);
    c->atlas_source = uv_atlas_strdup(atlas);
    c->vertex_source = uv_atlas_strdup(vertex);
    c->vertices = malloc(layout->byte_size);
    bool ok = c->a0_source && c->atlas_source && c->vertex_source && c->vertices;
    if (ok) { memcpy(c->vertices, vertices, layout->byte_size); c->layout = *layout; }
    else c->invalidated = true;
    c->result.source_abi_ok = ok;
    return uv_atlas_leave(c) && ok;
}

static bool uv_atlas_material(pl_gpu gpu, pl_tex target,
    const struct pl_opengl_uv_atlas_material *material)
{
    (void)gpu;
    struct pl_tex_gl *t=PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c=t->atlas_anchor;
    if (!c || !material || c->has_material || !uv_atlas_enter(c)) return false;
    memcpy(&c->material,material,sizeof(*material)); c->has_material=true;
    bool ok=!c->reader || !memcmp(material,&c->reader->cache->seed->material,sizeof(*material));
    if (!ok) {c->reader->cache->invalidated=true;c->reader->result.reject_reason=1;}
    return uv_atlas_leave(c) && ok;
}

static bool uv_atlas_input_identity(pl_gpu gpu, pl_tex target, const void *texture)
{
    (void)gpu;
    struct pl_tex_gl *t=PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c=t->atlas_anchor;
    if (!c || !texture || !uv_atlas_enter(c)) return false;
    c->input_texture_token=(uintptr_t)texture;
    if (!c->reader) {
        struct pl_tex_gl *source=PL_PRIV((pl_tex)texture);
        if (source->atlas_input_anchor && source->atlas_input_anchor!=c) {
            uv_atlas_leave(c);return false;
        }
        source->atlas_input_anchor=c;c->input_watch=(pl_tex)texture;
    }
    // The watched texture is renderer-internal with a single-frame lifetime
    // (freed at frame end, reallocated for the redraw), so pointer equality
    // with the seed token is structurally unsatisfiable on-device. The token
    // pair stays archived for identity review; same-frame admission is
    // carried by frame id, PTS, material signature and geometry instead.
    return uv_atlas_leave(c);
}

static bool uv_atlas_uv_corners(pl_gpu gpu, pl_tex target, const float uv[8])
{
    (void)gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c || !uv || c->has_seed_uv || !uv_atlas_enter(c))
        return false;
    memcpy(c->seed_uv, uv, sizeof(c->seed_uv));
    c->has_seed_uv = true;
    return uv_atlas_leave(c);
}

struct pl_uv_atlas_anchor *gl_tex_uv_atlas_anchor(pl_gpu gpu, pl_tex target)
{
    (void) gpu;
    struct pl_tex_gl *t = PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c = t->atlas_anchor;
    if (!c) return NULL;
    if (c->finished) return NULL; // transaction complete; ordinary dispatches must not disturb it
    if (c->consumed || !uv_atlas_enter(c)) {
        c->invalidated = true; c->dispatch.rejected = true;
        return &c->dispatch;
    }
    c->consumed = true;
    c->transaction_hold=true; // paired by finish, including dispatch error paths
    return &c->dispatch;
}

bool pl_opengl_uv_atlas_anchor_cancel(pl_opengl_uv_atlas_anchor *handle)
{
    if (!handle || !*handle) return true;
    pl_opengl_uv_atlas_anchor c = *handle;
    if (c->transaction_hold) return false; // caller must finish/abort the render first
    pl_gpu gpu = c->gpu;
    if (!MAKE_CURRENT()) { c->invalidated = true; return false; }
    if (!uv_atlas_owner(c)) { RELEASE_CURRENT(); return false; }
    if (c->target) {
        struct pl_tex_gl *t = PL_PRIV(c->target);
        if (t->atlas_anchor == c) t->atlas_anchor = NULL;
        c->target = NULL;
    }
    if (c->input_watch) {
        struct pl_tex_gl *input=PL_PRIV(c->input_watch);
        if (input->atlas_input_anchor==c) input->atlas_input_anchor=NULL;
        c->input_watch=NULL;
    }
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (c->fence) gl->DeleteSync(c->fence);
    pl_pass_destroy(gpu, &c->dispatch.companion);
    pl_pass_destroy(gpu, &c->dispatch.b0_pass);
    pl_buf_destroy(gpu, &c->dispatch.probe_storage);
    pl_buf_destroy(gpu, &c->dispatch.storage);
    free(c->rgb); free(c->atlas_cpu); free(c->a0_source); free(c->atlas_source);
    free(c->vertex_source); free(c->original_body); free(c->vertices);
    gl_check_err(gpu, "uv_atlas_cancel");
    RELEASE_CURRENT();
    free(c); *handle = NULL;
    return true;
}

pl_opengl_uv_atlas_anchor pl_opengl_uv_atlas_anchor_arm(pl_gpu gpu, pl_tex target,
    const struct pl_opengl_uv_atlas_anchor_params *params)
{
    if (!gpu || !target || !params || !params->capture_id || !params->generation || !pl_opengl_get(gpu)) return NULL;
    struct pl_gl *p = PL_PRIV(gpu);
    struct pl_tex_gl *t = PL_PRIV(target);
    const struct pl_tex_params *q = &target->params;
    size_t pixels = q->w > 0 && q->h > 0 &&
        (size_t)q->w <= PL_OPENGL_UV_ATLAS_ANCHOR_MAX_PIXELS / (size_t)q->h
        ? (size_t)q->w * q->h : 0;
    bool format = q->format && q->format->type == PL_FMT_FLOAT && q->format->num_components == 4;
    for (int n=0; format && n<4; n++) format &= q->format->component_depth[n] == 16;
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!format || !pixels || pixels > PL_OPENGL_UV_ATLAS_ANCHOR_MAX_PIXELS || q->d || !q->renderable ||
        !t->fbo || t->rgb16f_to_yuv || t->compare || t->atlas_anchor || p->gles_ver < 31 ||
        !p->yuv_compare_caps_ready || pixels * 12 > p->yuv_compare_max_ssbo || pixels > PTRDIFF_MAX / 48 ||
        !gl->ReadPixels || !gl->ReadBuffer || !gl->PixelStorei || !gl->GetFramebufferAttachmentParameteriv ||
        !gl->CheckFramebufferStatus || !gl->BindFramebuffer || !gl->GetShaderPrecisionFormat ||
        !gl->MapBufferRange || !gl->UnmapBuffer)
    {
        PL_ERR(gpu, "MKS-YUV-ATLAS-LIB event=rejected reason=arm-capacity-or-format");
        return NULL;
    }
    if (!MAKE_CURRENT()) return NULL;
    EGLDisplay display = eglGetCurrentDisplay ? eglGetCurrentDisplay() : EGL_NO_DISPLAY;
    EGLContext context = eglGetCurrentContext ? eglGetCurrentContext() : EGL_NO_CONTEXT;
    EGLSurface draw = eglGetCurrentSurface ? eglGetCurrentSurface(EGL_DRAW) : EGL_NO_SURFACE;
    EGLSurface read = eglGetCurrentSurface ? eglGetCurrentSurface(EGL_READ) : EGL_NO_SURFACE;
    uint64_t before = p->error_epoch;
    bool owner = display && context && draw && read && display == p->egl_dpy && context == p->egl_ctx;
    GLint blocks=0, bindings=0, range[2]={0}, precision=0;
    if (owner) {
        gl->GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &blocks);
        gl->GetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &bindings);
        gl->GetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &precision);
    }
    bool clean = owner && gl_check_err(gpu, "uv_atlas_arm");
    if (!owner || !clean || p->error_epoch != before || blocks < 1 || bindings < 1 ||
        precision < 23 || range[0] < 126 || range[1] < 127 || gl_test_ext(gpu, "GL_ARB_buffer_storage", 44, 0))
    {
        PL_ERR(gpu, "MKS-YUV-ATLAS-LIB event=rejected reason=arm-owner-or-limits");
        RELEASE_CURRENT(); return NULL;
    }
    pl_opengl_uv_atlas_anchor c = calloc(1, sizeof(*c));
    if (!c) { RELEASE_CURRENT(); return NULL; }
    c->gpu = gpu; c->target = target; c->fbo = t->fbo; c->entry_epoch = before;
    c->production_seed = params->production_seed;
    c->owner_display=display; c->owner_context=context; c->owner_draw=draw; c->owner_read=read;
    c->result = (struct pl_opengl_uv_atlas_anchor_result) {
        .capture_id=params->capture_id, .generation=params->generation,
        .width=q->w, .height=q->h, .pixels=pixels, .rgba_words=pixels*4, .atlas_words=pixels*3,
        .read_format=GL_RGBA, .read_type=GL_FLOAT,
    };
    // Production seeds run zero-readback: no triple-RGB staging buffer and
    // no long-lived CPU atlas mirror (the ~41MiB host copy was diagnostic
    // scan state). The GPU SSBO still starts zeroed: a small staging page is
    // uploaded in chunks right after creation.
    if (!params->production_seed) {
        c->rgb = malloc(pixels * 48);
        c->atlas_cpu = calloc(pixels * 3, sizeof(uint32_t));
    }
    c->dispatch = (struct pl_uv_atlas_anchor) {
        .width=q->w, .height=q->h, .body=uv_atlas_body, .begin=uv_atlas_begin,
        .checkpoint=uv_atlas_checkpoint, .record=uv_atlas_record,
        .readback=uv_atlas_readback, .finish=uv_atlas_finish,
        .material=uv_atlas_material, .input_identity=uv_atlas_input_identity,
        .uv_corners=uv_atlas_uv_corners,
    };
    GLint old=0;
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &old);
    if (c->rgb && c->atlas_cpu)
        c->dispatch.storage = gl_buf_create(gpu, pl_buf_params(
            .size=pixels*12, .storable=true, .initial_data=c->atlas_cpu));
    if (!c->dispatch.storage && params->production_seed) {
        c->dispatch.storage = gl_buf_create(gpu, pl_buf_params(
            .size=pixels*12, .storable=true));
        if (c->dispatch.storage) {
            const size_t total = pixels*12;
            const size_t page = 1u << 16;
            void *zeros = calloc(1, page);
            struct pl_buf_gl *zbuf = PL_PRIV(c->dispatch.storage);
            bool zeroed = zeros != NULL;
            if (zeroed) {
                gl->BindBuffer(GL_ARRAY_BUFFER, zbuf->buffer);
                for (size_t off = 0; off < total; off += page)
                    gl->BufferSubData(GL_ARRAY_BUFFER, zbuf->offset + off,
                                      PL_MIN(page, total - off), zeros);
                free(zeros);
            }
            if (!zeroed) {
                // Fail closed: an unzeroable table must never be armed.
                pl_buf_destroy(gpu, &c->dispatch.storage);
                c->dispatch.storage = NULL;
            }
        }
    }
    gl->BindBuffer(GL_ARRAY_BUFFER, old);
    clean = gl_check_err(gpu, "uv_atlas_allocate");
    bool bufs_ok = params->production_seed ? c->dispatch.storage != NULL
                                           : (c->rgb && c->atlas_cpu && c->dispatch.storage);
    bool ok = bufs_ok && clean && p->error_epoch == before && uv_atlas_owner(c);
    RELEASE_CURRENT();
    if (!ok) {
        PL_ERR(gpu, "MKS-YUV-ATLAS-LIB event=rejected reason=allocation-or-epoch");
        pl_opengl_uv_atlas_anchor_cancel(&c);
        return NULL;
    }
    t->atlas_anchor = c;
    return c;
}

bool pl_opengl_uv_atlas_anchor_baseline_ok(pl_opengl_uv_atlas_anchor c)
{
    if (!c || !c->finished || !c->baseline_ok || !uv_atlas_enter(c)) return false;
    return uv_atlas_leave(c);
}

enum pl_opengl_uv_atlas_anchor_status pl_opengl_uv_atlas_anchor_poll(
    pl_opengl_uv_atlas_anchor c, struct pl_opengl_uv_atlas_anchor_result *result)
{
    if (!c || !result || !uv_atlas_enter(c)) {
        if (c)
            PL_ERR(c->gpu, "MKS-ANCHOR-DIAG poll enter inv=%d", c->invalidated ? 1 : 0);
        return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
    }
    if (uv_atlas_sealed(c) && c->fence) {
        // The transaction's own cleanliness was verified at finish. Between
        // finish and this poll, the seeding frame's ordinary swap path (final
        // pass + eglSwapBuffers) may have left benign recoverable GL errors
        // pending; drain them — on the owner context enter() just verified —
        // so the completed anchor is not invalidated by work outside its
        // transaction. The fence stays the completion gate.
        const gl_funcs *gl0 = gl_funcs_get(c->gpu);
        while (gl0->GetError() != GL_NO_ERROR)
            ;
    }
    pl_gpu gpu = c->gpu;
    const gl_funcs *gl = gl_funcs_get(gpu);
    if (!c->finished) { *result=c->result; return uv_atlas_leave(c) ? PL_OPENGL_UV_ATLAS_ANCHOR_PENDING : PL_OPENGL_UV_ATLAS_ANCHOR_FAILED; }
    if (!c->baseline_ok || !c->fence) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG poll baseline=%d fence=%d inv=%d",
               c->baseline_ok ? 1 : 0, c->fence ? 1 : 0, c->invalidated ? 1 : 0);
        RELEASE_CURRENT(); return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
    }
    if (!c->completed) {
        GLenum ready = gl->ClientWaitSync(c->fence, 0, 0);
        if (ready == GL_TIMEOUT_EXPIRED) {
            // Pending is fine, but the wait's own error surface must be clean:
            // gl_check_err's return misses a callback that failed and bumped
            // the epoch, so compare the local epoch delta as well.
            struct pl_gl *p_wait = PL_PRIV(gpu);
            uint64_t wait_epoch = p_wait->error_epoch;
            bool clean = gl_check_err(gpu, "uv_atlas_poll_wait");
            if (!clean || p_wait->error_epoch != wait_epoch) {
                PL_ERR(gpu, "MKS-ANCHOR-DIAG poll wait-own-op clean=%d epoch=%llu/%llu",
                       clean ? 1 : 0, (unsigned long long) p_wait->error_epoch,
                       (unsigned long long) wait_epoch);
                c->invalidated=true;
                RELEASE_CURRENT();
                return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
            }
            *result=c->result;
            return uv_atlas_leave(c) ? PL_OPENGL_UV_ATLAS_ANCHOR_PENDING : PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
        }
        if (ready != GL_ALREADY_SIGNALED && ready != GL_CONDITION_SATISFIED) {
            PL_ERR(gpu, "MKS-ANCHOR-DIAG poll fence-wait ready=0x%x", (unsigned) ready);
            c->invalidated=true; RELEASE_CURRENT(); return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
        }
        struct pl_gl *p = PL_PRIV(gpu);
        if (!c->production_seed) {
            GLint old=0;
            gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &old);
            struct pl_buf_gl *buf = PL_PRIV(c->dispatch.storage);
            gl->BindBuffer(GL_ARRAY_BUFFER, buf->buffer);
            void *data = gl->MapBufferRange(GL_ARRAY_BUFFER, buf->offset, c->result.atlas_words*4, GL_MAP_READ_BIT);
            bool ok = data != NULL;
            if (data) {
                memcpy(c->atlas_cpu, data, c->result.atlas_words*4);
                ok = gl->UnmapBuffer(GL_ARRAY_BUFFER) == GL_TRUE;
            }
            gl->BindBuffer(GL_ARRAY_BUFFER, old);
            bool clean = gl_check_err(gpu, "uv_atlas_poll");
            if (!ok || !clean || p->error_epoch != c->entry_epoch) {
                c->invalidated=true; RELEASE_CURRENT(); return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
            }
        } else {
            // Sealed production seed: the drain above cleared any pending
            // external error, so this only fails on errors raised by the
            // poll's own work (fence wait, or a callback that ran inside
            // gl_check_err and failed — caught by the local epoch delta,
            // since unrelated rendering's drift is tolerated by sealing).
            uint64_t poll_epoch = p->error_epoch;
            bool clean = gl_check_err(gpu, "uv_atlas_poll");
            if (!clean || p->error_epoch != poll_epoch) {
                PL_ERR(gpu, "MKS-ANCHOR-DIAG poll own-op clean=%d epoch=%llu/%llu",
                       clean ? 1 : 0, (unsigned long long) p->error_epoch,
                       (unsigned long long) poll_epoch);
                c->invalidated=true; RELEASE_CURRENT(); return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
            }
        }
        if (!c->production_seed)
            for (size_t n=0; n<c->result.pixels; n++) {
                const uint32_t *v = c->atlas_cpu + n*3;
                c->result.bad_atlas_pixels += v[2] != 1 || (v[0]&0x7f800000u)==0x7f800000u || (v[1]&0x7f800000u)==0x7f800000u;
            }
        c->result.fence_signaled=true;
        c->result.atlas_complete = c->production_seed || c->result.bad_atlas_pixels == 0;
        if (c->production_seed)
            // Zero-readback seeding: coverage flags stay archived as observed,
            // and completion is the fence plus clean GL state alone. Replay
            // equality remains the diagnostic mode's contract.
            c->result.anchor_equal = true;
        else
            c->result.anchor_equal = c->result.atlas_complete && c->result.half_exact[0] && c->result.half_exact[1] &&
                c->result.half_exact[2] && !c->result.mismatches_before_atlas && !c->result.mismatches_before_after;
        c->completed=true;
    }
    *result=c->result;
    return uv_atlas_leave(c) ? PL_OPENGL_UV_ATLAS_ANCHOR_COMPLETE : PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
}

static bool uv_atlas_getter(pl_opengl_uv_atlas_anchor c)
{
    if (!c || !c->completed || !uv_atlas_enter(c)) return false;
    return uv_atlas_leave(c);
}

bool pl_opengl_uv_atlas_anchor_data(pl_opengl_uv_atlas_anchor c, const uint32_t **before,
    const uint32_t **atlas, const uint32_t **after, const uint32_t **words)
{
    if (!before || !atlas || !after || !words || !uv_atlas_getter(c)) return false;
    *before=c->rgb; *atlas=c->rgb+c->result.rgba_words; *after=c->rgb+c->result.rgba_words*2; *words=c->atlas_cpu;
    return true;
}

bool pl_opengl_uv_atlas_anchor_sources(pl_opengl_uv_atlas_anchor c, const char **a0,
    const char **atlas, const char **vertex, const char **body)
{
    if (!a0 || !atlas || !vertex || !body || !uv_atlas_getter(c)) return false;
    *a0=c->a0_source; *atlas=c->atlas_source; *vertex=c->vertex_source; *body=c->original_body;
    return *a0 && *atlas && *vertex && *body;
}

bool pl_opengl_uv_atlas_anchor_vertices(pl_opengl_uv_atlas_anchor c,
    const uint8_t **payload, struct pl_opengl_yuv_compare_quad_layout *layout)
{
    if (!payload || !layout || !uv_atlas_getter(c) || !c->vertices) return false;
    *payload=c->vertices; *layout=c->layout;
    return true;
}

// ---- Explicit promote/consume diagnostic, never automatic production reuse ----
pl_opengl_uv_atlas_cache pl_opengl_uv_atlas_promote(pl_opengl_uv_atlas_anchor *anchor)
{
    if (!anchor || !*anchor) return NULL;
    pl_opengl_uv_atlas_anchor c=*anchor;
    if (!c->completed || !c->result.anchor_equal || !c->has_material ||
        !c->input_texture_token || c->reader || c->transaction_hold || !uv_atlas_enter(c)) return NULL;
    pl_opengl_uv_atlas_cache cache=calloc(1,sizeof(*cache));
    if (!cache) {uv_atlas_leave(c);return NULL;}
    pl_gpu gpu=c->gpu;
    const gl_funcs *gl=gl_funcs_get(gpu);
    struct pl_buf_gl *b=PL_PRIV(c->dispatch.storage);
    cache->seed=c;
    cache->info=(struct pl_opengl_uv_atlas_cache_info){
        .cache_id=c->result.capture_id,.seed_capture_id=c->result.capture_id,
        .generation=c->result.generation,.width=c->result.width,.height=c->result.height,
        .pixels=c->result.pixels,.atlas_words=c->result.atlas_words,
        .storage_buffer=b->buffer,.storage_offset=b->offset,.storage_size=c->dispatch.storage->params.size,
    };
    // Move ownership before cleanup. Even a consumed cleanup error must not
    // leave two externally owned handles pointing at the same GPU allocation.
    if (c->target) {
        struct pl_tex_gl *t=PL_PRIV(c->target);
        if (t->atlas_anchor==c) t->atlas_anchor=NULL;
        c->target=NULL;
    }
    *anchor=NULL;
    if (c->fence) {gl->DeleteSync(c->fence);c->fence=NULL;}
    pl_pass_destroy(gpu,&c->dispatch.companion);
    free(c->rgb);c->rgb=NULL;
    free(c->atlas_source);c->atlas_source=NULL;
    bool clean=gl_check_err(gpu,"uv_atlas_promote");
    if (!clean) c->invalidated=true;
    if (!uv_atlas_leave(c)) cache->invalidated=true;
    return cache; // invalidated cache is still owned and must be destroyed
}

bool pl_opengl_uv_atlas_table_attach(pl_gpu gpu, pl_opengl_uv_atlas_cache cache)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!gpu || !cache || cache->invalidated || cache->active || !cache->seed ||
        !cache->seed->dispatch.storage || !cache->seed->has_material ||
        !cache->seed->vertices || !pl_opengl_get(gpu))
        return false;
    // The table's storage buffer belongs to the GPU that seeded it; binding
    // it on another GPU's context would reference foreign GL objects.
    if (cache->seed->gpu != gpu)
        return false;
    if (p->uv_atlas_production == cache)
        return true;
    if (p->uv_atlas_production)
        return false; // detach first; one live table at a time
    p->uv_atlas_production = cache;
    p->uv_atlas_production_generation++;
    return true;
}

void pl_opengl_uv_atlas_table_detach(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!gpu || !p->uv_atlas_production)
        return;
    p->uv_atlas_production = NULL;
    p->uv_atlas_production_generation++;
}

bool pl_opengl_uv_atlas_table_active(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    return gpu && p->uv_atlas_production && !p->uv_atlas_production->invalidated;
}

uint64_t pl_opengl_uv_atlas_table_pass_serial(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    return gpu ? p->uv_atlas_table_passes : 0;
}

void pl_opengl_uv_atlas_table_note_pass(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (gpu)
        p->uv_atlas_table_passes++;
}

pl_buf pl_opengl_uv_atlas_table_info(pl_gpu gpu, int *width, int *height,
                                     size_t *words, uint64_t *generation)
{
    struct pl_gl *p = PL_PRIV(gpu);
    pl_opengl_uv_atlas_cache cache = gpu ? p->uv_atlas_production : NULL;
    if (!cache || cache->invalidated || cache->seed->gpu != gpu) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG info-null reason=%s",
               !cache ? "detached" : cache->invalidated ? "invalidated" : "gpu-mismatch");
        return NULL;
    }
    pl_buf storage = cache->seed->dispatch.storage;
    if (!storage) {
        PL_ERR(gpu, "MKS-ANCHOR-DIAG info-null reason=no-storage");
        return NULL;
    }
    if (width)
        *width = cache->info.width;
    if (height)
        *height = cache->info.height;
    if (words)
        *words = (size_t) storage->params.size / sizeof(uint32_t);
    if (generation)
        *generation = p->uv_atlas_production_generation;
    return storage;
}

const struct pl_opengl_uv_atlas_material *pl_opengl_uv_atlas_table_seed_material(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    pl_opengl_uv_atlas_cache cache = gpu ? p->uv_atlas_production : NULL;
    if (!cache || cache->invalidated || cache->seed->gpu != gpu || !cache->seed->has_material)
        return NULL;
    return &cache->seed->material;
}

const float *pl_opengl_uv_atlas_table_seed_uv(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    pl_opengl_uv_atlas_cache cache = gpu ? p->uv_atlas_production : NULL;
    if (!cache || cache->invalidated || cache->seed->gpu != gpu || !cache->seed->has_seed_uv)
        return NULL;
    return cache->seed->seed_uv;
}

bool pl_opengl_uv_atlas_cache_info(pl_opengl_uv_atlas_cache cache,
    struct pl_opengl_uv_atlas_cache_info *info,const uint32_t **atlas)
{
    // Production seeds keep no CPU atlas mirror; the GPU table is the
    // product. Report no diagnostic view rather than a NULL "success".
    if (!cache || !info || !atlas || !uv_atlas_getter(cache->seed) ||
        !cache->seed->atlas_cpu) return false;
    *info=cache->info;*atlas=cache->seed->atlas_cpu;return true;
}

bool pl_opengl_uv_atlas_cache_destroy(pl_opengl_uv_atlas_cache *handle)
{
    if (handle && *handle) {
        pl_gpu gpu = (*handle)->seed->gpu;
        struct pl_gl *pg = PL_PRIV(gpu);
        if (pg->uv_atlas_production == *handle)
            pl_opengl_uv_atlas_table_detach(gpu);
    }
    if (!handle || !*handle) return true;
    pl_opengl_uv_atlas_cache cache=*handle;
    if (cache->active || !pl_opengl_uv_atlas_anchor_cancel(&cache->seed)) return false;
    free(cache);*handle=NULL;return true;
}

static bool uv_read_layout_equal(const struct pl_opengl_yuv_compare_quad_layout *a,
                                 const struct pl_opengl_yuv_compare_quad_layout *b)
{
    if (a->primitive!=b->primitive || a->indexed!=b->indexed || a->vertex_count!=b->vertex_count ||
        a->num_attributes!=b->num_attributes || a->position_attrib!=b->position_attrib ||
        a->stride!=b->stride || a->byte_size!=b->byte_size ||
        memcmp(&a->viewport,&b->viewport,sizeof(a->viewport)) ||
        memcmp(&a->scissors,&b->scissors,sizeof(a->scissors))) return false;
    for (int i=0;i<a->num_attributes;i++) {
        const struct pl_opengl_yuv_compare_quad_attrib *x=&a->attributes[i],*y=&b->attributes[i];
        if (strcmp(x->name,y->name) || strcmp(x->format_name,y->format_name) || strcmp(x->glsl_type,y->glsl_type) ||
            x->fmt_type!=y->fmt_type || x->location!=y->location || x->num_components!=y->num_components ||
            x->opaque!=y->opaque || x->num_planes!=y->num_planes || x->offset!=y->offset || x->texel_size!=y->texel_size ||
            memcmp(x->component_depth,y->component_depth,sizeof(x->component_depth)) ||
            memcmp(x->host_bits,y->host_bits,sizeof(x->host_bits)) ||
            memcmp(x->sample_order,y->sample_order,sizeof(x->sample_order))) return false;
    }
    return true;
}

static bool uv_read_begin(pl_gpu gpu,pl_tex target,const char *a0,const char *probe,
    const char *vertex,const struct pl_opengl_yuv_compare_quad_layout *layout,const uint8_t *vertices)
{
    struct pl_tex_gl *t=PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c=t->atlas_anchor;
    if (!c || !c->reader || !uv_atlas_enter(c)) return false;
    struct pl_opengl_uv_atlas_read_t *r=c->reader;
    pl_opengl_uv_atlas_anchor seed=r->cache->seed;
    bool ok=c->has_material && !r->cache->invalidated && a0 && probe && vertex && layout && vertices &&
        c->dispatch.b0_pass && !strcmp(a0,seed->a0_source) &&
        !strcmp(vertex,seed->vertex_source) && !strcmp(c->original_body,seed->original_body) &&
        uv_read_layout_equal(layout,&seed->layout) &&
        layout->byte_size==seed->layout.byte_size && !memcmp(vertices,seed->vertices,layout->byte_size);
    if (!ok) {r->result.reject_reason=1;r->cache->invalidated=true;uv_atlas_leave(c);return false;}
    r->b0_source=uv_atlas_strdup(c->dispatch.b0_pass->params.glsl_shader);
    bool clean=uv_atlas_leave(c);
    if (!clean || !r->b0_source || !uv_atlas_begin(gpu,target,a0,probe,vertex,layout,vertices)) return false;
    r->result.signature_match=true;
    r->result.b0_program_token=(uintptr_t)c->dispatch.b0_pass;
    return true;
}

static bool uv_read_probe(pl_gpu gpu,pl_tex target,unsigned int mode)
{
    struct pl_tex_gl *t=PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c=t->atlas_anchor;
    if (!c || !c->reader || mode>1 || !uv_atlas_enter(c)) return false;
    struct pl_opengl_uv_atlas_read_t *r=c->reader;
    unsigned stage=mode+1;
    struct pl_gl *p=PL_PRIV(gpu);
    uint64_t before=p->error_epoch;
    const gl_funcs *gl=gl_funcs_get(gpu);
    GLint old=0;
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING,&old);
    gl->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    struct pl_buf_gl *b=PL_PRIV(c->dispatch.probe_storage);
    gl->BindBuffer(GL_ARRAY_BUFFER,b->buffer);
    void *mapped=gl->MapBufferRange(GL_ARRAY_BUFFER,b->offset,r->result.probe_words*4,GL_MAP_READ_BIT);
    bool ok=mapped!=NULL;
    if (mapped) {
        memcpy(mode ? r->half : r->uv,mapped,r->result.probe_words*4);
        ok=gl->UnmapBuffer(GL_ARRAY_BUFFER)==GL_TRUE;
    }
    if (ok && mode==0) {
        // The second independent payload starts from zero without allocating
        // another GPU SSBO. Half CPU storage is still zero at this point.
        // This private allocation is verified mutable BufferData storage.
        // Do not use pl_buf_write: generic host_writable is deliberately off.
        gl->BufferSubData(GL_ARRAY_BUFFER,b->offset,r->result.probe_words*4,r->half);
    }
    gl->BindBuffer(GL_ARRAY_BUFFER,old);
    bool clean=gl_check_err(gpu,"uv_atlas_read_probe");
    ok &= clean && before==c->entry_epoch && p->error_epoch==before;
    c->result.readback_ok[stage]=ok;
    c->result.readback_epoch_before[stage]=before;
    c->result.readback_epoch_after[stage]=p->error_epoch;
    r->result.probe_modes[mode]=mode;
    if (!ok) c->invalidated=true;
    return uv_atlas_leave(c) && ok;
}

static void uv_read_finish(pl_gpu gpu,pl_tex target,bool ok)
{
    struct pl_tex_gl *t=PL_PRIV(target);
    pl_opengl_uv_atlas_anchor c=t->atlas_anchor;
    if (!c || !c->reader || c->finished) return;
    struct pl_opengl_uv_atlas_read_t *r=c->reader;
    if (!uv_atlas_enter(c)) {
        r->result.reject_reason=5;c->finished=true;c->dispatch.ended=true;
        PL_ERR(gpu,"MKS-YUV-ATLAS-LIB event=rejected reason=read-gl-or-owner");
        c->baseline_ok=false;uv_atlas_drop_hold(c);return;
    }
    struct pl_gl *p=PL_PRIV(gpu);const gl_funcs *gl=gl_funcs_get(gpu);
    bool a_ok=c->result.draw_ok[0] && c->result.readback_ok[0] && c->result.half_exact[0] &&
        !c->result.bad_half_components[0] && p->error_epoch==c->entry_epoch;
    for (unsigned i=0;i<3;i++) ok &= c->result.draw_ok[i] && c->result.readback_ok[i] &&
        c->result.draw_epoch_before[i]==c->entry_epoch && c->result.draw_epoch_after[i]==c->entry_epoch &&
        c->result.readback_epoch_before[i]==c->entry_epoch && c->result.readback_epoch_after[i]==c->entry_epoch;
    ok &= r->result.signature_match && r->result.b0_program_token &&
        c->result.program_tokens[1] && c->result.program_tokens[1]==c->result.program_tokens[2] &&
        c->result.program_tokens[1]!=r->result.b0_program_token;
    if (ok) {
        c->fence=gl->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);gl->Flush();
        bool clean=gl_check_err(gpu,"uv_atlas_read_finish");
        ok=c->fence && clean && p->error_epoch==c->entry_epoch;
        if (!ok) {c->invalidated=true;r->result.reject_reason=5;}
    }
    c->baseline_ok=a_ok && p->error_epoch==c->entry_epoch;
    c->finished=true;c->result.finish_epoch=p->error_epoch;
    r->result.baseline_ok=c->baseline_ok;
    if (!ok) {r->cache->invalidated=true;if (!r->result.reject_reason)
        r->result.reject_reason=c->dispatch.reader_reject ? c->dispatch.reader_reject : 2;}
    if (!uv_atlas_leave(c)) {c->baseline_ok=false;r->result.reject_reason=5;}
    c->dispatch.ended=true;uv_atlas_drop_hold(c);
    if (!ok) {
        static const char *reasons[]={"none","read-signature","read-unsupported","read-input-identity",
            "read-allocation-or-compile","read-gl-or-owner"};
        unsigned int code=r->result.reject_reason;
        PL_ERR(gpu,"MKS-YUV-ATLAS-LIB event=rejected reason=%s",reasons[code<=5 ? code : 5]);
    }
}

pl_opengl_uv_atlas_read pl_opengl_uv_atlas_read_arm(pl_opengl_uv_atlas_cache cache,
    pl_tex rgb,pl_tex yuv,const struct pl_opengl_uv_atlas_read_params *params)
{
    if (!cache || !params || cache->invalidated || cache->active || !rgb || !yuv || rgb==yuv ||
        !params->capture_id || params->generation!=cache->info.generation ||
        params->ordinal!=cache->completed_reads+1 || params->ordinal>2) return NULL;
    pl_opengl_uv_atlas_anchor seed=cache->seed;
    pl_gpu gpu=seed->gpu;
    struct pl_tex_gl *t=PL_PRIV(rgb), *yt=PL_PRIV(yuv);
    pl_fmt format=rgb->params.format;
    bool rgb16f=format && format->type==PL_FMT_FLOAT && format->num_components==4;
    for (int i=0;rgb16f && i<4;i++) rgb16f &= format->component_depth[i]==16;
    if (t->atlas_anchor || t->compare || t->rgb16f_to_yuv || yt->atlas_read_owner ||
        !yt->rgb16f_to_yuv || yt->compare || yt->atlas_anchor || !t->fbo || t->fbo==yt->fbo ||
        !rgb16f || !rgb->params.renderable || rgb->params.d || !yuv->params.format ||
        rgb->params.w!=cache->info.width || rgb->params.h!=cache->info.height ||
        yuv->params.w!=cache->info.width || yuv->params.h!=cache->info.height ||
        rgb->params.format->signature!=yuv->params.format->signature || !uv_atlas_enter(seed)) return NULL;
    const gl_funcs *gl=gl_funcs_get(gpu);struct pl_gl *p=PL_PRIV(gpu);
    GLint blocks=0,bindings=0;
    gl->GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,&blocks);
    gl->GetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS,&bindings);
    bool clean=gl_check_err(gpu,"uv_atlas_read_limits");
    size_t bytes=cache->info.pixels*16;
    bool left=uv_atlas_leave(seed);
    if (!clean || blocks<2 || bindings<2 || bytes>p->yuv_compare_max_ssbo ||
        !gl->GetIntegeri_v || !gl->BufferSubData || !left) return NULL;
    struct pl_opengl_uv_atlas_read_t *r=calloc(1,sizeof(*r));
    pl_opengl_uv_atlas_anchor c=calloc(1,sizeof(*c));
    if (!r || !c) {free(r);free(c);return NULL;}
    r->state=c;r->cache=cache;c->reader=r;c->gpu=gpu;c->target=rgb;c->fbo=t->fbo;
    c->entry_epoch=seed->entry_epoch;c->owner_display=seed->owner_display;c->owner_context=seed->owner_context;
    c->owner_draw=seed->owner_draw;c->owner_read=seed->owner_read;
    c->result=(struct pl_opengl_uv_atlas_anchor_result){.capture_id=params->capture_id,.generation=params->generation,
        .width=cache->info.width,.height=cache->info.height,.pixels=cache->info.pixels,
        .rgba_words=cache->info.pixels*4,.read_format=GL_RGBA,.read_type=GL_FLOAT};
    r->result=(struct pl_opengl_uv_atlas_read_result){.capture_id=params->capture_id,
        .cache_id=cache->info.cache_id,.seed_capture_id=cache->info.seed_capture_id,
        .generation=params->generation,.ordinal=params->ordinal,.width=cache->info.width,.height=cache->info.height,
        .pixels=cache->info.pixels,.rgba_words=cache->info.pixels*4,.probe_words=cache->info.pixels*4,
        .seed_input_texture_token=seed->input_texture_token};
    c->rgb=malloc(bytes);r->uv=calloc(1,bytes);r->half=calloc(1,bytes);
    c->dispatch=(struct pl_uv_atlas_anchor){.storage=seed->dispatch.storage,.reader=true,
        .yuv_target=yuv,.width=cache->info.width,.height=cache->info.height,
        .body=uv_atlas_body,.material=uv_atlas_material,.input_identity=uv_atlas_input_identity,
        .uv_corners=uv_atlas_uv_corners,
        .begin=uv_read_begin,.checkpoint=uv_atlas_checkpoint,.record=uv_atlas_record,
        .readback=uv_atlas_readback,.probe_readback=uv_read_probe,.finish=uv_read_finish};
    bool ok=c->rgb && r->uv && r->half && uv_atlas_enter(c);
    if (ok) {
        GLint old=0;gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING,&old);
        c->dispatch.probe_storage=gl_buf_create(gpu,pl_buf_params(.size=bytes,.storable=true,.initial_data=r->uv));
        gl->BindBuffer(GL_ARRAY_BUFFER,old);
        clean=gl_check_err(gpu,"uv_atlas_read_allocate");
        left=uv_atlas_leave(c);
        struct pl_buf_gl *allocated=c->dispatch.probe_storage ? PL_PRIV(c->dispatch.probe_storage) : NULL;
        struct pl_buf_gl *seed_buffer=PL_PRIV(seed->dispatch.storage);
        ok=allocated && allocated->buffer!=seed_buffer->buffer && clean && left;
    }
    if (!ok) {
        if (!c->dispatch.probe_storage) {
            // No owned GL resource exists yet, so CPU cleanup is legal even
            // if the make_current callback changed the EGL owner.
            free(c->rgb);free(r->uv);free(r->half);free(c);free(r);
        } else {
            pl_opengl_uv_atlas_read_cancel(&r); // retain/leak rather than delete foreign GL
        }
        return NULL;
    }
    cache->active=r;t->atlas_anchor=c;yt->atlas_read_owner=r;
    return r;
}

bool pl_opengl_uv_atlas_read_baseline_ok(pl_opengl_uv_atlas_read r)
{return r && pl_opengl_uv_atlas_anchor_baseline_ok(r->state);}

static enum pl_opengl_uv_atlas_anchor_status uv_read_return(pl_opengl_uv_atlas_read r,
    struct pl_opengl_uv_atlas_read_result *result,enum pl_opengl_uv_atlas_anchor_status status)
{
    if (!uv_atlas_leave(r->state)) {
        r->result.reject_reason=5;r->result.baseline_ok=false;r->cache->invalidated=true;
        status=PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
    }
    *result=r->result;return status;
}

enum pl_opengl_uv_atlas_anchor_status pl_opengl_uv_atlas_read_poll(
    pl_opengl_uv_atlas_read r,struct pl_opengl_uv_atlas_read_result *result)
{
    if (!r || !result) return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
    pl_opengl_uv_atlas_anchor c=r->state;
    if (!uv_atlas_enter(c)) {
        r->result.reject_reason=5;r->result.baseline_ok=false;r->cache->invalidated=true;
        *result=r->result;return PL_OPENGL_UV_ATLAS_ANCHOR_FAILED;
    }
    r->result.input_texture_token=c->input_texture_token;
    r->result.finish_epoch=c->result.finish_epoch;r->result.baseline_ok=c->baseline_ok;
    for (unsigned i=0;i<3;i++) {
        r->result.draw_ok[i]=c->result.draw_ok[i];r->result.readback_ok[i]=c->result.readback_ok[i];
        r->result.program_tokens[i]=c->result.program_tokens[i];
        r->result.draw_epoch_before[i]=c->result.draw_epoch_before[i];r->result.draw_epoch_after[i]=c->result.draw_epoch_after[i];
        r->result.readback_epoch_before[i]=c->result.readback_epoch_before[i];r->result.readback_epoch_after[i]=c->result.readback_epoch_after[i];
    }
    if (!c->finished) return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_PENDING);
    if (r->result.reject_reason || !c->baseline_ok || !c->fence)
        return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_FAILED);
    if (!c->completed) {
        const gl_funcs *gl=gl_funcs_get(c->gpu);
        GLenum ready=gl->ClientWaitSync(c->fence,0,0);
        bool clean=gl_check_err(c->gpu,"uv_atlas_read_wait");
        struct pl_gl *p=PL_PRIV(c->gpu);
        if (!clean || p->error_epoch!=c->entry_epoch) {c->invalidated=true;return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_FAILED);}
        if (ready==GL_TIMEOUT_EXPIRED) return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_PENDING);
        if (ready!=GL_ALREADY_SIGNALED && ready!=GL_CONDITION_SATISFIED) {c->invalidated=true;return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_FAILED);}
        for (size_t n=0;n<c->result.pixels;n++) {
            const uint32_t *uv=r->uv+n*4,*half=r->half+n*4,*seed=r->cache->seed->atlas_cpu+n*3;
            r->result.bad_uv_pixels += uv[0]!=1 || uv[3]!=1 || uv[1]!=seed[0] || uv[2]!=seed[1] ||
                (uv[1]&0x7f800000u)==0x7f800000u || (uv[2]&0x7f800000u)==0x7f800000u;
            bool bad=half[0]!=1;
            for (int k=0;k<3;k++) bad |= !mks_atlas_half_exact(half[1+k]) || half[1+k]!=c->rgb[n*4+k];
            r->result.bad_half_pixels+=bad;
        }
        r->result.uv_equal=!r->result.bad_uv_pixels;r->result.half_equal=!r->result.bad_half_pixels;
        r->result.fence_signaled=true;c->completed=true;
        if (r->result.uv_equal && r->result.half_equal) {
            r->cache->completed_reads++;
            if (r->result.ordinal==1 && r->cache->seed->input_watch) {
                pl_opengl_uv_atlas_anchor seed=r->cache->seed;
                struct pl_tex_gl *input=PL_PRIV(seed->input_watch);
                if (input->atlas_input_anchor==seed) input->atlas_input_anchor=NULL;
                seed->input_watch=NULL;
            }
        }
        else r->cache->invalidated=true;
    }
    return uv_read_return(r,result,PL_OPENGL_UV_ATLAS_ANCHOR_COMPLETE);
}

bool pl_opengl_uv_atlas_read_data(pl_opengl_uv_atlas_read r,const uint32_t **a0,const uint32_t **uv,const uint32_t **half)
{
    if (!r || !a0 || !uv || !half || !uv_atlas_getter(r->state)) return false;
    *a0=r->state->rgb;*uv=r->uv;*half=r->half;return true;
}
bool pl_opengl_uv_atlas_read_sources(pl_opengl_uv_atlas_read r,const char **a0,const char **b0,
    const char **probe,const char **vertex,const char **body)
{
    if (!r || !a0 || !b0 || !probe || !vertex || !body || !uv_atlas_getter(r->state)) return false;
    *a0=r->state->a0_source;*b0=r->b0_source;*probe=r->state->atlas_source;
    *vertex=r->state->vertex_source;*body=r->state->original_body;return true;
}
bool pl_opengl_uv_atlas_read_signature(pl_opengl_uv_atlas_read r,
    const struct pl_opengl_uv_atlas_material **seed,const struct pl_opengl_uv_atlas_material **current,
    const uint8_t **vertices,struct pl_opengl_yuv_compare_quad_layout *layout)
{
    if (!r || !seed || !current || !vertices || !layout || !uv_atlas_getter(r->state)) return false;
    *seed=&r->cache->seed->material;*current=&r->state->material;
    *vertices=r->state->vertices;*layout=r->state->layout;return true;
}
bool pl_opengl_uv_atlas_read_cancel(pl_opengl_uv_atlas_read *handle)
{
    if (!handle || !*handle) return true;
    struct pl_opengl_uv_atlas_read_t *r=*handle;
    pl_opengl_uv_atlas_anchor c=r->state;
    if (c->transaction_hold) return false;
    pl_gpu gpu=c->gpu;
    if (!MAKE_CURRENT()) return false;
    if (!uv_atlas_owner(c)) {RELEASE_CURRENT();return false;}
    if (c->dispatch.yuv_target) {
        struct pl_tex_gl *yt=PL_PRIV(c->dispatch.yuv_target);
        if (yt->atlas_read_owner==r) yt->atlas_read_owner=NULL;
        c->dispatch.yuv_target=NULL;
    }
    RELEASE_CURRENT();
    c->dispatch.storage=NULL; // immutable seed owns this allocation
    if (!pl_opengl_uv_atlas_anchor_cancel(&r->state)) return false;
    if (r->cache->active==r) r->cache->active=NULL;
    free(r->uv);free(r->half);free(r->b0_source);free(r);*handle=NULL;return true;
}
