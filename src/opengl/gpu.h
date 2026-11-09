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

#pragma once

#include "../gpu.h"
#include "common.h"

// Thread safety: Unsafe, same as pl_gpu_destroy
pl_gpu pl_gpu_create_gl(pl_log log, pl_opengl gl, const struct pl_opengl_params *params);

// --- pl_gpu internal structs and functions

struct pl_gl {
    struct pl_gpu_fns impl;
    pl_opengl gl;
    uint64_t sig; // hash of GL_VERSION, GL_VENDOR, GL_RENDERER
    bool failed;
    uint64_t error_epoch;
    bool yuv_compare_caps_ready;
    size_t yuv_compare_max_ssbo;
    // Default-off production UV atlas table (see opengl.h). Null = disabled;
    // a generation counter invalidates cached consumer passes on re-attach.
    struct pl_opengl_uv_atlas_cache_t *uv_atlas_production;
    uint64_t uv_atlas_production_generation;
    // Monotonic count of passes that actually sampled through the table;
    // the caller verifies it advances on every table-driven frame.
    uint64_t uv_atlas_table_passes;

    // For import/export
    EGLDisplay egl_dpy;
    EGLContext egl_ctx;
    bool egl_storage;
#ifdef PL_HAVE_UNIX
    // List of formats supported by EGL_EXT_image_dma_buf_import
    PL_ARRAY(EGLint) egl_formats;
#endif

    // Sync objects and associated callbacks
    PL_ARRAY(struct gl_cb) callbacks;


    // Incrementing counters to keep track of object uniqueness
    int buf_id;

    // Cached capabilities
    int gl_ver;
    int gles_ver;
    bool has_storage;
    bool has_invalidate_fb;
    bool has_invalidate_tex;
    bool has_vao;
    bool has_queries;
    bool mks_timers_disabled;
    // Android opt-in EXT timer bridge; never enabled by default.
    bool ext_queries;
    uint64_t timer_epoch;
    uint64_t timer_issued, timer_resolved, timer_disjoint, timer_dropped;
    pl_clock_t timer_report_time;
    void (*timer_gen)(GLsizei, GLuint *);
    void (*timer_delete)(GLsizei, const GLuint *);
    void (*timer_begin)(GLenum, GLuint);
    void (*timer_end)(GLenum);
    void (*timer_info)(GLenum, GLenum, GLint *);
    void (*timer_get)(GLuint, GLenum, GLuint *);
    void (*timer_get64)(GLuint, GLenum, uint64_t *);
    bool has_modifiers;
    bool has_readback;
    bool has_egl_storage;
    bool has_egl_import;
    int gather_comps;
};

static inline const gl_funcs *gl_funcs_get(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    struct gl_ctx *glctx = PL_PRIV(p->gl);
    return &glctx->func;
}

void gl_timer_begin(pl_gpu gpu, pl_timer timer);
void gl_timer_end(pl_gpu gpu, pl_timer timer);

static inline bool _make_current(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    if (!gl_make_current(p->gl)) {
        p->failed = true;
        p->error_epoch++;
        return false;
    }

    return true;
}

static inline void _release_current(pl_gpu gpu)
{
    struct pl_gl *p = PL_PRIV(gpu);
    gl_release_current(p->gl);
}

#define MAKE_CURRENT() _make_current(gpu)
#define RELEASE_CURRENT() _release_current(gpu)

struct gl_yuv_compare {
    struct pl_yuv_compare dispatch;
    struct pl_opengl_yuv_compare_params params;
    struct pl_opengl_yuv_compare_result result;
    GLsync fence;
    uint64_t finish_epoch;
    size_t storage_words;
    bool last_baseline_ok;
    EGLDisplay owner_display;
    EGLContext owner_context;
    EGLSurface owner_draw, owner_read;
    char *candidate_source, *baseline_source, *vertex_source;
    char *original_body, *tap_body, *candidate_vertex_source;
    struct pl_opengl_yuv_compare_quad_layout quad_layout;
    uint8_t *quad_baseline_data, *quad_candidate_data;
    bool quad_recorded;
    bool consumed, finished, invalidated;
};
struct pl_tex_gl {
    struct gl_yuv_compare *compare;
    struct pl_opengl_uv_atlas_anchor_t *atlas_anchor;
    struct pl_opengl_uv_atlas_read_t *atlas_read_owner;
    struct pl_opengl_uv_atlas_anchor_t *atlas_input_anchor;
    bool rgb16f_to_yuv;
    uint64_t yuv_successful_draws;
    GLenum target;
    GLuint texture;
    bool wrapped_tex;
    GLuint fbo; // or 0
    bool wrapped_fb;
    GLbitfield barrier;

    // GL format fields
    GLenum format;
    GLint iformat;
    GLenum type;

    // For imported/exported textures
    EGLImageKHR image;
    int fd;
};

pl_tex gl_tex_create(pl_gpu, const struct pl_tex_params *);
void gl_tex_destroy(pl_gpu, pl_tex);
void gl_tex_invalidate(pl_gpu, pl_tex);
void gl_tex_clear_ex(pl_gpu, pl_tex, const union pl_clear_color);
void gl_tex_blit(pl_gpu, const struct pl_tex_blit_params *);
bool gl_tex_upload(pl_gpu, const struct pl_tex_transfer_params *);
bool gl_tex_download(pl_gpu, const struct pl_tex_transfer_params *);

struct pl_buf_gl {
    uint64_t id; // unique per buffer
    GLuint buffer;
    size_t offset;
    GLsync fence;
    GLbitfield barrier;
    bool mapped;
};

pl_buf gl_buf_create(pl_gpu, const struct pl_buf_params *);
void gl_buf_destroy(pl_gpu, pl_buf);
void gl_buf_write(pl_gpu, pl_buf, size_t offset, const void *src, size_t size);
bool gl_buf_read(pl_gpu, pl_buf, size_t offset, void *dst, size_t size);
void gl_buf_copy(pl_gpu, pl_buf dst, size_t dst_offset,
                 pl_buf src, size_t src_offset, size_t size);
bool gl_buf_poll(pl_gpu, pl_buf, uint64_t timeout);

struct pl_pass_gl;
int gl_desc_namespace(pl_gpu, enum pl_desc_type type);
pl_pass gl_pass_create(pl_gpu, const struct pl_pass_params *);
void gl_pass_destroy(pl_gpu, pl_pass);
void gl_pass_run(pl_gpu, const struct pl_pass_run_params *);

struct pl_yuv_compare *gl_tex_yuv_compare(pl_gpu gpu, pl_tex tex);

struct pl_uv_atlas_anchor *gl_tex_uv_atlas_anchor(pl_gpu gpu, pl_tex tex);
