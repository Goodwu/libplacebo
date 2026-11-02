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

#include "common.h"
#include "log.h"
#include "shaders.h"
#include "sample_tap.h"
#include "atlas_anchor.h"
#include "atlas_read.h"
#include <libplacebo/config.h> // PL_HAVE_OPENGL: the UV table is GL-only
#include <libplacebo/opengl.h> // explicit GL quad diagnostic payload ABI
#include "dispatch.h"
#include "gpu.h"
#include "pl_thread.h"

// Maximum number of passes to keep around at once. If full, passes older than
// MIN_AGE are evicted to make room. (Failing that, the passes array doubles)
#define MAX_PASSES 100
#define MIN_AGE 10

enum {
    TMP_PRELUDE,   // GLSL version, global definitions, etc.
    TMP_MAIN,      // main GLSL shader body
    TMP_VERT_HEAD, // vertex shader inputs/outputs
    TMP_VERT_BODY, // vertex shader body
    TMP_COUNT,
};

struct pl_dispatch_t {
    pl_mutex lock;
    pl_log log;
    pl_gpu gpu;
    uint8_t current_ident;
    uint8_t current_index;
    bool dynamic_constants;
    int max_passes;

    void (*info_callback)(void *, const struct pl_dispatch_info *);
    void *info_priv;

    PL_ARRAY(pl_shader) shaders;                // to avoid re-allocations
    PL_ARRAY(struct pass *) passes;             // compiled passes

    // temporary buffers to help avoid re_allocations during pass creation
    PL_ARRAY(const struct pl_buffer_var *) buf_tmp;
    pl_str_builder tmp[TMP_COUNT];
    uint8_t *ubo_tmp;
};

enum pass_var_type {
    PASS_VAR_NONE = 0,
    PASS_VAR_GLOBAL, // regular/global uniforms
    PASS_VAR_UBO,    // uniform buffers
    PASS_VAR_PUSHC   // push constants
};

// Cached metadata about a variable's effective placement / update method
struct pass_var {
    int index; // for pl_var_update
    enum pass_var_type type;
    struct pl_var_layout layout;
    void *cached_data;
};

struct pass {
    uint64_t signature;
    pl_pass pass;
    pl_pass compare_baseline; // explicit diagnostic companion, same input layout
    uint64_t compare_body_hash, compare_tap_body_hash;
    int compare_binding, compare_desc_index, compare_oracle_var;
    int compare_probe_binding, compare_probe_desc_index, compare_probe_pair_var;
    char *compare_sources[3], *compare_bodies[2];
    char *compare_vertices[2];
    char compare_tap_varying[8];
    int last_index;
    const void *owned_vertex_data;
    size_t owned_vertex_bytes;
    int position_attrib;

    // contains cached data and update metadata, same order as pl_shader
    struct pass_var *vars;
    int num_var_locs;

    // for uniform buffer updates
    struct pl_shader_desc ubo_desc; // temporary
    int ubo_index;
    pl_buf ubo;

    // Cached pl_pass_run_params. This will also contain mutable allocations
    // for the push constants, descriptor bindings (including the binding for
    // the UBO pre-filled), vertex array and variable updates
    struct pl_pass_run_params run_params;

    // for pl_dispatch_info
    pl_timer timer;
    uint64_t ts_last;
    uint64_t ts_peak;
    uint64_t ts_sum;
    uint64_t samples[PL_ARRAY_SIZE(((struct pl_dispatch_info *) NULL)->samples)];
    int ts_idx;
};

static void pass_destroy(pl_dispatch dp, struct pass *pass)
{
    if (!pass)
        return;

    pl_buf_destroy(dp->gpu, &pass->ubo);
    pl_pass_destroy(dp->gpu, &pass->pass);
    pl_pass_destroy(dp->gpu, &pass->compare_baseline);
    pl_timer_destroy(dp->gpu, &pass->timer);
    pl_free(pass);
}

pl_dispatch pl_dispatch_create(pl_log log, pl_gpu gpu)
{
    struct pl_dispatch_t *dp = pl_zalloc_ptr(NULL, dp);
    pl_mutex_init(&dp->lock);
    dp->log = log;
    dp->gpu = gpu;
    dp->max_passes = MAX_PASSES;
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp); i++)
        dp->tmp[i] = pl_str_builder_alloc(dp);

    return dp;
}

void pl_dispatch_destroy(pl_dispatch *ptr)
{
    pl_dispatch dp = *ptr;
    if (!dp)
        return;

    for (int i = 0; i < dp->passes.num; i++)
        pass_destroy(dp, dp->passes.elem[i]);
    for (int i = 0; i < dp->shaders.num; i++)
        pl_shader_free(&dp->shaders.elem[i]);

    pl_mutex_destroy(&dp->lock);
    pl_free(dp);
    *ptr = NULL;
}

pl_shader pl_dispatch_begin_ex(pl_dispatch dp, bool unique)
{
    pl_mutex_lock(&dp->lock);

    struct pl_shader_params params = {
        .id = unique ? dp->current_ident++ : 0,
        .gpu = dp->gpu,
        .index = dp->current_index,
        .dynamic_constants = dp->dynamic_constants,
    };

    pl_shader sh = NULL;
    PL_ARRAY_POP(dp->shaders, &sh);
    pl_mutex_unlock(&dp->lock);

    if (sh) {
        pl_shader_reset(sh, &params);
        return sh;
    }

    return pl_shader_alloc(dp->log, &params);
}

void pl_dispatch_mark_dynamic(pl_dispatch dp, bool dynamic)
{
    dp->dynamic_constants = dynamic;
}

void pl_dispatch_callback(pl_dispatch dp, void *priv,
                          void (*cb)(void *priv, const struct pl_dispatch_info *))
{
    dp->info_callback = cb;
    dp->info_priv = priv;
}

pl_shader pl_dispatch_begin(pl_dispatch dp)
{
    return pl_dispatch_begin_ex(dp, false);
}

static bool add_pass_var(pl_dispatch dp, void *tmp, struct pass *pass,
                         struct pl_pass_params *params,
                         const struct pl_shader_var *sv, struct pass_var *pv,
                         bool greedy)
{
    pl_gpu gpu = dp->gpu;
    if (pv->type)
        return true;

    // Try not to use push constants for "large" values like matrices in the
    // first pass, since this is likely to exceed the VGPR/pushc size budgets
    bool try_pushc = greedy || (sv->var.dim_m == 1 && sv->var.dim_a == 1) || sv->dynamic;
    if (try_pushc && gpu->glsl.vulkan && gpu->limits.max_pushc_size) {
        pv->layout = pl_std430_layout(params->push_constants_size, &sv->var);
        size_t new_size = pv->layout.offset + pv->layout.size;
        if (new_size <= gpu->limits.max_pushc_size) {
            params->push_constants_size = new_size;
            pv->type = PASS_VAR_PUSHC;
            return true;
        }
    }

    // If we haven't placed all PCs yet, don't place anything else, since
    // we want to try and fit more stuff into PCs before "giving up"
    if (!greedy)
        return true;

    int num_locs = sv->var.dim_v * sv->var.dim_m * sv->var.dim_a;
    bool can_var = pass->num_var_locs + num_locs <= gpu->limits.max_variable_comps;

    // Attempt using uniform buffer next. The GLSL version 440 check is due
    // to explicit offsets on UBO entries. In theory we could leave away
    // the offsets and support UBOs for older GL as well, but this is a nice
    // safety net for driver bugs (and also rules out potentially buggy drivers)
    // Also avoid UBOs for highly dynamic stuff since that requires synchronizing
    // the UBO writes every frame
    bool try_ubo = !can_var || !sv->dynamic;
    if (try_ubo && gpu->glsl.version >= 440 && gpu->limits.max_ubo_size) {
        if (sh_buf_desc_append(tmp, gpu, &pass->ubo_desc, &pv->layout, sv->var)) {
            pv->type = PASS_VAR_UBO;
            return true;
        }
    }

    // Otherwise, use global uniforms
    if (can_var) {
        pv->type = PASS_VAR_GLOBAL;
        pv->index = params->num_variables;
        pv->layout = pl_var_host_layout(0, &sv->var);
        PL_ARRAY_APPEND_RAW(tmp, params->variables, params->num_variables, sv->var);
        pass->num_var_locs += num_locs;
        return true;
    }

    // Ran out of variable binding methods. The most likely scenario in which
    // this can happen is if we're using a GPU that does not support global
    // input vars and we've exhausted the UBO size limits.
    PL_ERR(dp, "Unable to add input variable: possibly exhausted "
           "variable count / UBO size limits?");
    return false;
}

#define ADD(b, ...)     pl_str_builder_addf(b, __VA_ARGS__)
#define ADD_CAT(b, cat) pl_str_builder_concat(b, cat)
#define ADD_CONST(b, s) pl_str_builder_const_str(b, s)

static void add_var(pl_str_builder body, const struct pl_var *var)
{
    const char *type = pl_var_glsl_type_name(*var);
    if (var->dim_a > 1) {
        ADD(body, "%s "$"[%d];\n", type, sh_ident_unpack(var->name), var->dim_a);
    } else {
        ADD(body, "%s "$";\n", type, sh_ident_unpack(var->name));
    }
}

static int cmp_buffer_var(const void *pa, const void *pb)
{
    const struct pl_buffer_var * const *a = pa, * const *b = pb;
    return PL_CMP((*a)->layout.offset, (*b)->layout.offset);
}

static void add_buffer_vars(pl_dispatch dp, void *tmp, pl_str_builder body,
                            const struct pl_buffer_var *vars, int num)
{
    // Sort buffer vars by offset
    PL_ARRAY_RESIZE(dp, dp->buf_tmp, num);
    for (int i = 0; i < num; i++)
        dp->buf_tmp.elem[i] = &vars[i];
    qsort(dp->buf_tmp.elem, num, sizeof(&vars[0]), cmp_buffer_var);

    ADD(body, "{\n");
    for (int i = 0; i < num; i++) {
        const struct pl_buffer_var *bv = dp->buf_tmp.elem[i];
        // Add an explicit offset wherever possible
        if (dp->gpu->glsl.version >= 440)
            ADD(body, "    layout(offset=%zu) ", bv->layout.offset);
        add_var(body, &bv->var);
    }
    ADD(body, "};\n");
}

struct generate_params {
    void *tmp;
    pl_shader sh;
    struct pass *pass;
    struct pl_pass_params *pass_params;
    ident_t out_mat;
    ident_t out_off;
    int vert_idx;
    bool rgb16f_to_yuv;
    const struct pl_yuv_compare *compare;
    ident_t compare_words;
    pl_str_builder shader_body;
    // Default-off production UV table: non-NULL storage rewrites the direct
    // sample coordinate to a readonly SSBO lookup (schema10-validated shape).
    pl_buf uv_table_storage;
    int uv_table_width;
    size_t uv_table_words;
    ident_t uv_table_ident;
    ident_t uv_table_words_ident;
};

static void generate_shaders(pl_dispatch dp,
                             const struct generate_params *params,
                             pl_str_builder *out_vert_builder,
                             pl_str_builder *out_glsl_builder)
{
    pl_gpu gpu = dp->gpu;
    pl_shader sh = params->sh;
    void *tmp = params->tmp;
    struct pass *pass = params->pass;
    struct pl_pass_params *pass_params = params->pass_params;
    pl_str_builder shader_body = params->shader_body;

    pl_str_builder pre = dp->tmp[TMP_PRELUDE];
    ADD(pre, "#version %d%s\n", gpu->glsl.version,
        (gpu->glsl.gles && gpu->glsl.version > 100) ? " es" : "");
    if (params->rgb16f_to_yuv)
        ADD(pre, "#extension GL_EXT_YUV_target : require\n");
    if (pass_params->type == PL_PASS_COMPUTE)
        ADD(pre, "#extension GL_ARB_compute_shader : enable\n");

    // Enable this unconditionally if the GPU supports it, since we have no way
    // of knowing whether subgroups are being used or not
    if (gpu->glsl.subgroup_size) {
        ADD(pre, "#extension GL_KHR_shader_subgroup_basic : enable \n"
                 "#extension GL_KHR_shader_subgroup_vote : enable \n"
                 "#extension GL_KHR_shader_subgroup_arithmetic : enable \n"
                 "#extension GL_KHR_shader_subgroup_ballot : enable \n"
                 "#extension GL_KHR_shader_subgroup_shuffle : enable \n"
                 "#extension GL_KHR_shader_subgroup_clustered : enable \n"
                 "#extension GL_KHR_shader_subgroup_quad : enable \n");
    }

    // Enable all extensions needed for different types of input
    bool has_ssbo = false, has_ubo = false, has_img = false, has_texel = false,
         has_ext = false, has_ext_yuv = false, has_nofmt = false,
         has_gather = false;
    for (int i = 0; i < sh->descs.num; i++) {
        switch (sh->descs.elem[i].desc.type) {
        case PL_DESC_BUF_UNIFORM: has_ubo = true; break;
        case PL_DESC_BUF_STORAGE: has_ssbo = true; break;
        case PL_DESC_BUF_TEXEL_UNIFORM: has_texel = true; break;
        case PL_DESC_BUF_TEXEL_STORAGE: {
            pl_buf buf = sh->descs.elem[i].binding.object;
            has_nofmt |= !buf->params.format->glsl_format;
            has_texel = true;
            break;
        }
        case PL_DESC_STORAGE_IMG: {
            pl_tex tex = sh->descs.elem[i].binding.object;
            has_nofmt |= !tex->params.format->glsl_format;
            has_img = true;
            break;
        }
        case PL_DESC_SAMPLED_TEX: {
            pl_tex tex = sh->descs.elem[i].binding.object;
            has_gather |= tex->params.format->gatherable;
            switch (tex->sampler_type) {
            case PL_SAMPLER_NORMAL: break;
            case PL_SAMPLER_RECT: break;
            case PL_SAMPLER_EXTERNAL: has_ext = true; break;
            case PL_SAMPLER_EXTERNAL_YUV: has_ext_yuv = true; break;
            case PL_SAMPLER_TYPE_COUNT: pl_unreachable();
            }
            break;
        }

        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }

    if (has_img && !gpu->glsl.gles)
        ADD(pre, "#extension GL_ARB_shader_image_load_store : enable\n");
    if (has_ubo)
        ADD(pre, "#extension GL_ARB_uniform_buffer_object : enable\n");
    // Capture on GLES 3.1+ uses core SSBO declarations. Keep the ordinary
    // extension prelude unchanged outside this explicit diagnostic variant.
    if (has_ssbo && !(params->compare && gpu->glsl.gles && gpu->glsl.version >= 310))
        ADD(pre, "#extension GL_ARB_shader_storage_buffer_object : enable\n");
    if (has_texel)
        ADD(pre, "#extension GL_ARB_texture_buffer_object : enable\n");
    if (has_ext) {
        if (gpu->glsl.version >= 300) {
            ADD(pre, "#extension GL_OES_EGL_image_external_essl3 : enable\n");
        } else {
            ADD(pre, "#extension GL_OES_EGL_image_external : enable\n");
        }
    }
    if (has_ext_yuv)
        ADD(pre, "#extension GL_EXT_YUV_target : require\n");
    if (has_nofmt)
        ADD(pre, "#extension GL_EXT_shader_image_load_formatted : enable\n");
    if (has_gather && !gpu->glsl.gles)
        ADD(pre, "#extension GL_ARB_texture_gather : enable\n");

    if (gpu->glsl.gles) {
        // Use 32-bit precision for floats if possible
        ADD(pre, "#ifdef GL_FRAGMENT_PRECISION_HIGH \n"
                 "precision highp float;            \n"
                 "#else                             \n"
                 "precision mediump float;          \n"
                 "#endif                            \n");

        // Always use 16-bit precision for samplers
        ADD(pre, "precision mediump sampler2D; \n");
        if (gpu->limits.max_tex_1d_dim)
            ADD(pre, "precision mediump sampler1D; \n");
        if (gpu->limits.max_tex_3d_dim && gpu->glsl.version > 100)
            ADD(pre, "precision mediump sampler3D; \n");

        // Raw Y/Cb/Cr samplers benefit from full precision
        if (has_ext_yuv)
            ADD(pre, "precision highp __samplerExternal2DY2YEXT; \n");

        // Integer math has a good chance of caring about precision
        ADD(pre, "precision highp int; \n");
    }

    // textureLod() doesn't work on external/rect samplers, simply disable
    // LOD sampling in this case. We don't currently support mipmaps anyway.
    for (int i = 0; i < sh->descs.num; i++) {
        if (pass_params->descriptors[i].type != PL_DESC_SAMPLED_TEX)
            continue;
        pl_tex tex = sh->descs.elem[i].binding.object;
        if (tex->sampler_type != PL_SAMPLER_NORMAL) {
            ADD(pre, "#define textureLod(t, p, b) texture(t, p) \n"
                     "#define textureLodOffset(t, p, b, o)    \\\n"
                     "        textureOffset(t, p, o)            \n");
            break;
        }
    }

    // Add all of the push constants as their own element
    if (pass_params->push_constants_size) {
        // We re-use add_buffer_vars to make sure variables are sorted, this
        // is important because the push constants can be out-of-order in
        // `pass->vars`
        PL_ARRAY(struct pl_buffer_var) pc_bvars = {0};
        for (int i = 0; i < sh->vars.num; i++) {
            if (pass->vars[i].type != PASS_VAR_PUSHC)
                continue;

            PL_ARRAY_APPEND(tmp, pc_bvars, (struct pl_buffer_var) {
                .var = sh->vars.elem[i].var,
                .layout = pass->vars[i].layout,
            });
        }

        ADD(pre, "layout(std430, push_constant) uniform PushC ");
        add_buffer_vars(dp, tmp, pre, pc_bvars.elem, pc_bvars.num);
    }

    // Add all of the specialization constants
    for (int i = 0; i < sh->consts.num; i++) {
        static const char *types[PL_VAR_TYPE_COUNT] = {
            [PL_VAR_SINT]   = "int",
            [PL_VAR_UINT]   = "uint",
            [PL_VAR_FLOAT]  = "float",
        };

        const struct pl_shader_const *sc = &sh->consts.elem[i];
        ADD(pre, "layout(constant_id=%"PRIu32") const %s "$" = 1; \n",
            pass_params->constants[i].id, types[sc->type],
            sh_ident_unpack(sc->name));
    }

    static const char sampler_prefixes[PL_FMT_TYPE_COUNT] = {
        [PL_FMT_FLOAT]  = ' ',
        [PL_FMT_UNORM]  = ' ',
        [PL_FMT_SNORM]  = ' ',
        [PL_FMT_UINT]   = 'u',
        [PL_FMT_SINT]   = 'i',
    };

    // Add all of the required descriptors
    for (int i = 0; i < sh->descs.num; i++) {
        const struct pl_shader_desc *sd = &sh->descs.elem[i];
        const struct pl_desc *desc = &pass_params->descriptors[i];

        switch (desc->type) {
        case PL_DESC_SAMPLED_TEX: {
            static const char *types[][4] = {
                [PL_SAMPLER_NORMAL][1]  = "sampler1D",
                [PL_SAMPLER_NORMAL][2]  = "sampler2D",
                [PL_SAMPLER_NORMAL][3]  = "sampler3D",
                [PL_SAMPLER_RECT][2]    = "sampler2DRect",
                [PL_SAMPLER_EXTERNAL][2] = "samplerExternalOES",
                [PL_SAMPLER_EXTERNAL_YUV][2] = "__samplerExternal2DY2YEXT",
            };

            pl_tex tex = sd->binding.object;
            int dims = pl_tex_params_dimension(tex->params);
            const char *type = types[tex->sampler_type][dims];
            char prefix = sampler_prefixes[tex->params.format->type];
            ident_t id = sh_ident_unpack(desc->name);
            pl_assert(type && prefix);

            // Vulkan requires explicit bindings; GL always sets the
            // bindings manually to avoid relying on the user doing so
            if (gpu->glsl.vulkan) {
                ADD(pre, "layout(binding=%d) uniform %c%s "$";\n",
                    desc->binding, prefix, type, id);
            } else if (gpu->glsl.gles && prefix != ' ') {
                ADD(pre, "uniform highp %c%s "$";\n", prefix, type, id);
            } else {
                ADD(pre, "uniform %c%s "$";\n", prefix, type, id);
            }
            break;
        }

        case PL_DESC_STORAGE_IMG: {
            static const char *types[] = {
                [1] = "image1D",
                [2] = "image2D",
                [3] = "image3D",
            };

            // For better compatibility, we have to explicitly label the
            // type of data we will be reading/writing to this image.
            pl_tex tex = sd->binding.object;
            const char *format = tex->params.format->glsl_format;
            int dims = pl_tex_params_dimension(tex->params);
            if (gpu->glsl.vulkan) {
                if (format) {
                    ADD(pre, "layout(binding=%d, %s) ", desc->binding, format);
                } else {
                    ADD(pre, "layout(binding=%d) ", desc->binding);
                }
            } else if (format) {
                ADD(pre, "layout(%s) ", format);
            }

            ADD_CONST(pre, pl_desc_access_glsl_name(desc->access));
            if (sd->memory & PL_MEMORY_COHERENT)
                ADD(pre, " coherent");
            if (sd->memory & PL_MEMORY_VOLATILE)
                ADD(pre, " volatile");
            ADD(pre, " restrict uniform %s "$";\n",
                types[dims], sh_ident_unpack(desc->name));
            break;
        }

        case PL_DESC_BUF_UNIFORM:
            if (gpu->glsl.vulkan) {
                ADD(pre, "layout(std140, binding=%d) ", desc->binding);
            } else {
                ADD(pre, "layout(std140) ");
            }
            ADD(pre, "uniform "$" ", sh_ident_unpack(desc->name));
            add_buffer_vars(dp, tmp, pre, sd->buffer_vars, sd->num_buffer_vars);
            break;

        case PL_DESC_BUF_STORAGE:
            if (gpu->glsl.version >= 140)
                ADD(pre, "layout(std430, binding=%d) ", desc->binding);
            ADD_CONST(pre, pl_desc_access_glsl_name(desc->access));
            if (sd->memory & PL_MEMORY_COHERENT)
                ADD(pre, " coherent");
            if (sd->memory & PL_MEMORY_VOLATILE)
                ADD(pre, " volatile");
            ADD(pre, " restrict buffer "$" ", sh_ident_unpack(desc->name));
            add_buffer_vars(dp, tmp, pre, sd->buffer_vars, sd->num_buffer_vars);
            break;

        case PL_DESC_BUF_TEXEL_UNIFORM: {
            pl_buf buf = sd->binding.object;
            char prefix = sampler_prefixes[buf->params.format->type];
            if (gpu->glsl.vulkan)
                ADD(pre, "layout(binding=%d) ", desc->binding);
            ADD(pre, "uniform %csamplerBuffer "$";\n", prefix,
                sh_ident_unpack(desc->name));
            break;
        }

        case PL_DESC_BUF_TEXEL_STORAGE: {
            pl_buf buf = sd->binding.object;
            const char *format = buf->params.format->glsl_format;
            char prefix = sampler_prefixes[buf->params.format->type];
            if (gpu->glsl.vulkan) {
                if (format) {
                    ADD(pre, "layout(binding=%d, %s) ", desc->binding, format);
                } else {
                    ADD(pre, "layout(binding=%d) ", desc->binding);
                }
            } else if (format) {
                ADD(pre, "layout(%s) ", format);
            }

            ADD_CONST(pre, pl_desc_access_glsl_name(desc->access));
            if (sd->memory & PL_MEMORY_COHERENT)
                ADD(pre, " coherent");
            if (sd->memory & PL_MEMORY_VOLATILE)
                ADD(pre, " volatile");
            ADD(pre, " restrict uniform %cimageBuffer "$";\n",
                prefix, sh_ident_unpack(desc->name));
            break;
        }

        case PL_DESC_INVALID:
        case PL_DESC_TYPE_COUNT:
            pl_unreachable();
        }
    }

    // Add all of the remaining variables
    for (int i = 0; i < sh->vars.num; i++) {
        const struct pl_var *var = &sh->vars.elem[i].var;
        const struct pass_var *pv = &pass->vars[i];
        if (pv->type != PASS_VAR_GLOBAL)
            continue;
        ADD(pre, "uniform ");
        add_var(pre, var);
    }

    pl_str_builder glsl = dp->tmp[TMP_MAIN];
    ADD_CAT(glsl, pre);

    switch(pass_params->type) {
    case PL_PASS_RASTER: {
        pl_assert(params->vert_idx >= 0);
        pl_str_builder vert_head = dp->tmp[TMP_VERT_HEAD];
        pl_str_builder vert_body = dp->tmp[TMP_VERT_BODY];

        // Older GLSL doesn't support the use of explicit locations
        bool has_loc = gpu->glsl.version >= 430;

        // Set up a trivial vertex shader
        ADD_CAT(vert_head, pre);
        ADD(vert_body, "void main() {\n");
        for (int i = 0; i < sh->vas.num; i++) {
            const struct pl_vertex_attrib *va = &pass_params->vertex_attribs[i];
            const struct pl_shader_va *sva = &sh->vas.elem[i];
            const char *type = va->fmt->glsl_type;

            // Use the pl_shader_va for the name in the fragment shader since
            // the pl_vertex_attrib is already mangled for the vertex shader
            ident_t id = sh_ident_unpack(sva->attr.name);

            if (has_loc) {
                ADD(vert_head, "layout(location=%d) in %s "$";\n",
                    va->location, type, sh_ident_unpack(va->name));
            } else {
                ADD(vert_head, "in %s "$";\n", type, sh_ident_unpack(va->name));
            }

            if (i == params->vert_idx) {
                pl_assert(va->fmt->num_components == 2);
                ADD(vert_body, "vec2 va_pos = "$"; \n", sh_ident_unpack(va->name));
                if (params->out_mat)
                    ADD(vert_body, "va_pos = "$" * va_pos; \n", params->out_mat);
                if (params->out_off)
                    ADD(vert_body, "va_pos += "$"; \n", params->out_off);
                ADD(vert_body, "gl_Position = vec4(va_pos, 0.0, 1.0); \n");
            } else {
                // Everything else is just blindly passed through
                if (has_loc) {
                    ADD(vert_head, "layout(location=%d) out %s "$";\n",
                        va->location, type, id);
                    ADD(glsl, "layout(location=%d) in %s "$";\n",
                        va->location, type, id);
                } else {
                    ADD(vert_head, "out %s "$";\n", type, id);
                    ADD(glsl, "in %s "$";\n", type, id);
                }
                ADD(vert_body, $" = "$";\n", id, sh_ident_unpack(va->name));
            }
        }

        ADD(vert_body, "}");
        ADD_CAT(vert_head, vert_body);
        pl_hash_merge(&pass->signature, pl_str_builder_hash(vert_head));
        *out_vert_builder = vert_head;

        if (params->uv_table_storage && params->uv_table_ident) {
            // Production UV table rewrite. The macro rewrites exactly the
            // direct-sample coordinate identifier recorded by
            // pl_shader_sample_direct, reproducing the schema10-validated B0
            // substitution semantics. The SSBO block itself is declared once
            // by the sh_desc buffer-var emission (framework-assigned binding);
            // this helper indexes that block's single uint array member.
            pl_assert(params->uv_table_width > 0 && params->uv_table_words >= 3);
            ADD(glsl,
                "highp vec2 mks_cache_uv() {\n"
                "uvec2 mks_cache_p=uvec2(gl_FragCoord.xy);\n"
                "uint mks_cache_i=(mks_cache_p.y*%du+mks_cache_p.x)*3u;\n"
                "highp vec2 mks_cache_loaded=uintBitsToFloat(uvec2("
                $"[mks_cache_i]," $"[mks_cache_i+1u]));\n"
                "return mks_cache_loaded;\n}\n"
                "#define " $ " mks_cache_uv()\n",
                (unsigned) params->uv_table_width,
                params->uv_table_words_ident, params->uv_table_words_ident,
                params->uv_table_ident);
        }

        if (params->rgb16f_to_yuv) {
            ADD(glsl, "layout(yuv) out highp vec3 out_color;\n");
        } else if (has_loc) {
            ADD(glsl, "layout(location=0) out vec4 out_color;\n");
        } else {
            ADD(glsl, "out vec4 out_color;\n");
        }
        break;
    }
    case PL_PASS_COMPUTE:
        ADD(glsl, "layout (local_size_x = %d, local_size_y = %d) in;\n",
            sh->group_size[0], sh->group_size[1]);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    // Set up the main shader body
    ADD_CAT(glsl, shader_body);
    if (params->rgb16f_to_yuv) {
        // This exact small-value hypothesis passed the LG synthetic half-store
        // gate. It remains experimental until actual YUV/media equivalence.
        ADD(glsl,
            "highp float mks_half_store(highp float value) {\n"
            "highp uint raw=floatBitsToUint(value); highp uint mag=raw&0x7fffffffu;\n"
            "if(mag==0u||mag>=0x38800000u) return unpackHalf2x16(packHalf2x16(vec2(value,0.0))).x;\n"
            "highp uint sign=raw&0x80000000u; highp int e=int(mag>>23u)-127;\n"
            "if(e< -24) return uintBitsToFloat(sign);\n"
            "highp uint h=((mag&0x7fffffu)|0x800000u)>>uint(-e-1);\n"
            "highp float q=float(h)*0.000000059604644775390625;\n"
            "return uintBitsToFloat(floatBitsToUint(q)|sign); }\n");
    }
    ADD(glsl, "void main() {\n");
    if (params->compare && params->compare->sample_tap)
        ADD(glsl, "mks_tap_uv=vec2(0.0); mks_tap_sample=vec4(0.0); mks_tap_count=0u;\n");

    pl_assert(sh->input == PL_SHADER_SIG_NONE);
    switch (pass_params->type) {
    case PL_PASS_RASTER:
        pl_assert(sh->output == PL_SHADER_SIG_COLOR);
        if (params->rgb16f_to_yuv) {
            ADD(glsl, "highp vec3 mks_rgb = "$"().rgb;\n", sh->name);
            if (params->compare)
                ADD(glsl, "highp vec3 mks_pre=mks_rgb;\n");
            ADD(glsl,
                "mks_rgb=vec3(mks_half_store(mks_rgb.r),mks_half_store(mks_rgb.g),mks_half_store(mks_rgb.b));\n"
                "highp float mks_y=0.2627*mks_rgb.r+0.6780*mks_rgb.g+0.0593*mks_rgb.b;\n"
                "highp float mks_cb=(mks_rgb.b-mks_y)*0.5/(1.0-0.0593);\n"
                "highp float mks_cr=(mks_rgb.r-mks_y)*0.5/(1.0-0.2627);\n"
                "out_color=vec3(mks_y,mks_cb+0.5,mks_cr+0.5);\n");
        } else {
            ADD(glsl, "out_color = "$"();\n", sh->name);
        }
        if (params->compare) {
            const struct pl_yuv_compare *c = params->compare;
            ADD(glsl, "ivec2 mks_p=ivec2(gl_FragCoord.xy)-ivec2(%d,%d);\n"
                "if(all(greaterThanEqual(mks_p,ivec2(0)))&&all(lessThan(mks_p,ivec2(%d,%d)))){\n"
                "uint mks_i=uint(mks_p.y*%d+mks_p.x)*%uu;\n",
                c->x, c->y, c->width, c->height, c->width, c->words_per_pixel);
            const int count = params->rgb16f_to_yuv ? 0 : 10;
            ADD(glsl, "uint mks_old=atomicAdd("$"[mks_i+%du],1u);\nif(mks_old==0u){\n",
                params->compare_words, count);
            if (params->rgb16f_to_yuv) {
                for (int k=0; k<3; k++) {
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(mks_pre[%d]);\n", params->compare_words, 1+k, k);
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(mks_rgb[%d]);\n", params->compare_words, 4+k, k);
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(out_color[%d]);\n", params->compare_words, 7+k, k);
                }
            } else {
                for (int k=0; k<3; k++)
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(out_color[%d]);\n", params->compare_words, 11+k, k);
            }
            if (c->sample_tap) {
                int base = params->rgb16f_to_yuv ? 14 : 20;
                for (int k = 0; k < 2; k++)
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(mks_tap_uv[%d]);\n",
                        params->compare_words, base+k, k);
                for (int k = 0; k < 3; k++)
                    ADD(glsl, $"[mks_i+%du]=floatBitsToUint(mks_tap_sample[%d]);\n",
                        params->compare_words, base+2+k, k);
                ADD(glsl, $"[mks_i+%du]=mks_tap_count;\n", params->compare_words, base+5);
            }
            if (c->uv_oracle_probe && params->rgb16f_to_yuv) {
                ADD(glsl, "}\nuint mks_probe_i=(uint(mks_probe_pair)*%uu+uint(mks_p.y*%d+mks_p.x))*10u;\n"
                    "uint mks_probe_old=atomicAdd(mks_probe_words[mks_probe_i],1u);\n"
                    "if(mks_probe_old==0u){\n"
                    "mks_probe_words[mks_probe_i+1u]=mks_probe_helpers;\n"
                    "mks_probe_words[mks_probe_i+2u]=mks_probe_sites;\n"
                    "mks_probe_words[mks_probe_i+3u]=mks_probe_seen_counts.x;\n"
                    "mks_probe_words[mks_probe_i+4u]=mks_probe_seen_counts.y;\n"
                    "mks_probe_words[mks_probe_i+5u]=mks_probe_seen_uv.x;\n"
                    "mks_probe_words[mks_probe_i+6u]=mks_probe_seen_uv.y;\n"
                    "mks_probe_words[mks_probe_i+7u]=mks_probe_reason;\n"
                    "mks_probe_words[mks_probe_i+8u]=mks_oracle_valid ? 1u : 0u;\n"
                    "mks_probe_words[mks_probe_i+9u]=mks_probe_selected;\n}}\n",
                    (unsigned int) c->width * c->height, c->width);
            } else {
                ADD(glsl, "}}\n");
            }
        }
        break;
    case PL_PASS_COMPUTE:
        ADD(glsl, $"();\n", sh->name);
        break;
    case PL_PASS_INVALID:
    case PL_PASS_TYPE_COUNT:
        pl_unreachable();
    }

    ADD(glsl, "}");

    pl_hash_merge(&pass->signature, pl_str_builder_hash(glsl));
    *out_glsl_builder = glsl;
}

#undef ADD
#undef ADD_CAT

#define pass_age(pass) (dp->current_index - (pass)->last_index)

static int cmp_pass_age(const void *ptra, const void *ptrb)
{
    const struct pass *a = *(const struct pass **) ptra;
    const struct pass *b = *(const struct pass **) ptrb;
    return b->last_index - a->last_index;
}

static void garbage_collect_passes(pl_dispatch dp)
{
    if (dp->passes.num <= dp->max_passes)
        return;

    // Garbage collect oldest passes, starting at the middle
    qsort(dp->passes.elem, dp->passes.num, sizeof(struct pass *), cmp_pass_age);
    int idx = dp->passes.num / 2;
    while (idx < dp->passes.num && pass_age(dp->passes.elem[idx]) < MIN_AGE)
        idx++;

    for (int i = idx; i < dp->passes.num; i++)
        pass_destroy(dp, dp->passes.elem[i]);

    int num_evicted = dp->passes.num - idx;
    dp->passes.num = idx;

    if (num_evicted) {
        PL_DEBUG(dp, "Evicted %d passes from dispatch cache, consider "
                 "using more dynamic shaders", num_evicted);
    } else {
        dp->max_passes *= 2;
    }
}

struct pl_uv_atlas_table_params {
    pl_buf storage;
    int width;
    size_t words;
    uint64_t generation;
    ident_t words_ident;
};

static struct pass *finalize_pass(pl_dispatch dp, pl_shader sh,
                                  pl_tex target, int vert_idx,
                                  const struct pl_blend_params *blend, bool load,
                                  const struct pl_dispatch_vertex_params *vparams,
                                  const pl_transform2x2 *proj,
                                  const struct pl_yuv_compare *compare,
                                  ident_t compare_words, struct pl_uv_atlas_anchor *atlas,
                                  const struct pl_uv_atlas_table_params *uv_table)
{
    struct pass *pass = pl_alloc_ptr(dp, pass);
    *pass = (struct pass) {
        .signature = 0x0, // updated incrementally below
        .compare_binding = -1, .compare_desc_index = -1, .compare_oracle_var = -1,
        .compare_probe_binding = -1, .compare_probe_desc_index = -1, .compare_probe_pair_var = -1,
        .last_index = dp->current_index,
        .position_attrib = vert_idx,
        .ubo_desc = {
            .desc = {
                .name = sh_ident_pack(sh_fresh(sh, "UBO")),
                .type = PL_DESC_BUF_UNIFORM,
            },
        },
    };

    // For identifiers tied to the lifetime of this shader
    void *tmp = sh->tmp;

    struct pl_pass_params params = {
        .type = pl_shader_is_compute(sh) ? PL_PASS_COMPUTE : PL_PASS_RASTER,
        .num_descriptors = sh->descs.num,
        .vertex_type = vparams ? vparams->vertex_type : PL_PRIM_TRIANGLE_STRIP,
        .vertex_stride = vparams ? vparams->vertex_stride : 0,
        .blend_params = blend,
    };

    struct generate_params gen_params = {
        .tmp = tmp,
        .pass = pass,
        .pass_params = &params,
        .sh = sh,
        .vert_idx = vert_idx,
        .rgb16f_to_yuv = target && pl_tex_rgb16f_to_yuv(dp->gpu, target),
        .compare = compare, .compare_words = compare_words,
    };
    if (uv_table && uv_table->storage) {
        // The rewrite ident is resolved after the body is finalized, by the
        // same sample-tap discovery the schema10 validation used — the
        // production fusion pass samples through hooks whose coordinate is
        // not recorded by the fixed sampler entry points.
        gen_params.uv_table_storage = uv_table->storage;
        gen_params.uv_table_width = uv_table->width;
        gen_params.uv_table_words = uv_table->words;
        gen_params.uv_table_words_ident = uv_table->words_ident;
    }

    if (params.type == PL_PASS_RASTER) {
        assert(target);
        params.target_format = target->params.format;
        params.load_target = load;

        // Fill in the vertex attributes array
        params.num_vertex_attribs = sh->vas.num;
        params.vertex_attribs = pl_calloc_ptr(tmp, sh->vas.num, params.vertex_attribs);

        int va_loc = 0;
        for (int i = 0; i < sh->vas.num; i++) {
            struct pl_vertex_attrib *va = &params.vertex_attribs[i];
            *va = sh->vas.elem[i].attr;

            // Mangle the name to make sure it doesn't conflict with the
            // fragment shader input, this will be converted back to a legal
            // string by the shader compilation code
            va->name = sh_ident_pack(sh_fresh(sh, "va"));

            // Place the vertex attribute
            va->location = va_loc;
            if (!vparams) {
                va->offset = params.vertex_stride;
                params.vertex_stride += va->fmt->texel_size;
            }

            // The number of vertex attribute locations consumed by a vertex
            // attribute is the number of vec4s it consumes, rounded up
            const size_t va_loc_size = sizeof(float[4]);
            va_loc += PL_DIV_UP(va->fmt->texel_size, va_loc_size);
        }

        // Hash in the raster state configuration
        pl_hash_merge(&pass->signature, (uint64_t) params.vertex_type);
        pl_hash_merge(&pass->signature, (uint64_t) params.vertex_stride);
        pl_hash_merge(&pass->signature, (uint64_t) params.load_target);
        pl_hash_merge(&pass->signature, target->params.format->signature);
        if (gen_params.rgb16f_to_yuv)
            pl_hash_merge(&pass->signature, UINT64_C(0x59465556313646));
        if (gen_params.uv_table_storage) {
            pl_hash_merge(&pass->signature, UINT64_C(0x43503555564154));
            pl_hash_merge(&pass->signature, uv_table->generation);
        }
        if (blend) {
            pl_static_assert(sizeof(*blend) == sizeof(enum pl_blend_mode) * 4);
            pl_hash_merge(&pass->signature, pl_var_hash(*blend));
        }

        // Load projection matrix if required
        if (proj && memcmp(&proj->mat, &pl_matrix2x2_identity, sizeof(proj->mat)) != 0) {
            gen_params.out_mat = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_mat2("proj"),
                .data = PL_TRANSPOSE_2X2(proj->mat.m),
            });
        }

        if (proj && (proj->c[0] || proj->c[1])) {
            gen_params.out_off = sh_var(sh, (struct pl_shader_var) {
                .var = pl_var_vec2("offset"),
                .data = proj->c,
            });
        }
    }

    // Place all of the compile-time constants
    uint8_t *constant_data = NULL;
    if (sh->consts.num) {
        params.num_constants = sh->consts.num;
        params.constants = pl_alloc(tmp, sh->consts.num * sizeof(struct pl_constant));

        // Compute offsets
        size_t total_size = 0;
        uint32_t const_id = 0;
        for (int i = 0; i < sh->consts.num; i++) {
            params.constants[i] = (struct pl_constant) {
                .type = sh->consts.elem[i].type,
                .id = const_id++,
                .offset = total_size,
            };
            total_size += pl_var_type_size(sh->consts.elem[i].type);
        }

        // Write values into the constants buffer
        params.constant_data = constant_data = pl_alloc(pass, total_size);
        for (int i = 0; i < sh->consts.num; i++) {
            const struct pl_shader_const *sc = &sh->consts.elem[i];
            void *data = constant_data + params.constants[i].offset;
            memcpy(data, sc->data, pl_var_type_size(sc->type));
        }
    }

    // Place all the variables; these will dynamically end up in different
    // locations based on what the underlying GPU supports (UBOs, pushc, etc.)
    //
    // We go through the list twice, once to place stuff that we definitely
    // want inside PCs, and then a second time to opportunistically place the rest.
    pass->vars = pl_calloc_ptr(pass, sh->vars.num, pass->vars);
    for (int i = 0; i < sh->vars.num; i++) {
        if (!add_pass_var(dp, tmp, pass, &params, &sh->vars.elem[i], &pass->vars[i], false))
            goto error;
    }
    for (int i = 0; i < sh->vars.num; i++) {
        if (!add_pass_var(dp, tmp, pass, &params, &sh->vars.elem[i], &pass->vars[i], true))
            goto error;
    }

    // Now that we know the variable placement, finalize pushc/UBO sizes
    params.push_constants_size = PL_ALIGN2(params.push_constants_size, 4);
    size_t ubo_size = sh_buf_desc_size(&pass->ubo_desc);
    if (ubo_size) {
        pass->ubo_index = sh->descs.num;
        PL_ARRAY_APPEND(sh, sh->descs, pass->ubo_desc); // don't mangle names
    };

    // Place and fill in the descriptors
    const int num_descs = sh->descs.num;
    int binding[PL_DESC_TYPE_COUNT] = {0};
    params.num_descriptors = num_descs;
    params.descriptors = pl_calloc_ptr(tmp, num_descs, params.descriptors);
    for (int i = 0; i < num_descs; i++) {
        struct pl_desc *desc = &params.descriptors[i];
        *desc = sh->descs.elem[i].desc;
        desc->binding = binding[pl_desc_namespace(dp->gpu, desc->type)]++;
        if (compare && sh->descs.elem[i].binding.object == compare->storage &&
            desc->type == PL_DESC_BUF_STORAGE) {
            pass->compare_binding = desc->binding;
            pass->compare_desc_index = i;
        }
    }

    if (compare && compare->uv_oracle_probe)
        pass->compare_probe_binding = binding[pl_desc_namespace(dp->gpu, PL_DESC_BUF_STORAGE)];

    // Finalize the shader and look it up in the pass cache
    pl_str_builder vert_builder = NULL, glsl_builder = NULL;
    // Finalize upstream exactly once; both variants use this immutable body.
    gen_params.shader_body = sh_finalize_internal(sh);
    // Freeze the real ordinary pass before any capture descriptor/body change.
    if (atlas) {
        pl_str body = pl_str_builder_exec(gen_params.shader_body);
        char *original_body = pl_strdup0(tmp, body);
        if (compare || !atlas->body(dp->gpu, target, original_body) || !atlas->shader_body)
            goto error;
    }
    pass->compare_body_hash = compare ? pl_str_builder_hash(gen_params.shader_body) : 0;
    pl_str_builder a0_body = NULL;
    if (compare && compare->sample_tap) {
        pl_str body = pl_str_builder_exec(gen_params.shader_body);
        char *original = pl_strdup0(tmp, body);
        struct mks_sample_tap_site site;
        bool supported = dp->gpu->glsl.gles && dp->gpu->glsl.version >= 310 &&
                         params.type == PL_PASS_RASTER && mks_sample_tap_find(original, &site);
        unsigned int matching_tex = 0, matching_uv = 0, sampled_tex = 0;
        if (supported) {
            for (int i = 0; i < sh->descs.num; i++) {
                const struct pl_shader_desc *sd = &sh->descs.elem[i];
                if (sd->desc.type != PL_DESC_SAMPLED_TEX)
                    continue;
                sampled_tex++;
                pl_tex tex = sd->binding.object;
                if (sh_ident_unpack(sd->desc.name) == site.tex_id && tex &&
                    tex->sampler_type == PL_SAMPLER_NORMAL && tex->params.w > 0 &&
                    tex->params.h > 0 && !tex->params.d && tex->params.format &&
                    (tex->params.format->type == PL_FMT_FLOAT ||
                     tex->params.format->type == PL_FMT_UNORM ||
                     tex->params.format->type == PL_FMT_SNORM))
                    matching_tex++;
            }
            for (int i = 0; i < sh->vas.num; i++) {
                const struct pl_vertex_attrib *va = &sh->vas.elem[i].attr;
                if (sh_ident_unpack(va->name) == site.uv_id && va->fmt &&
                    va->fmt->type == PL_FMT_FLOAT && va->fmt->num_components == 2 &&
                    va->fmt->glsl_type && !strcmp(va->fmt->glsl_type, "vec2"))
                    matching_uv++;
            }
        }
        char *tap = supported && sampled_tex == 1 && matching_tex == 1 && matching_uv == 1
                    ? mks_sample_tap_instrument(original, &site) : NULL;
        if (!tap) {
            PL_ERR(dp, "MKS-YUV-COMPARE-LIB event=sample-tap-rejected reason=site-not-unique-or-supported");
            goto error;
        }
        memcpy(pass->compare_tap_varying, site.uv, sizeof(site.uv));
        pass->compare_bodies[0] = pl_strdup0(pass, pl_str0(original));
        pass->compare_bodies[1] = pl_strdup0(pass, pl_str0(tap));
        gen_params.shader_body = pl_str_builder_alloc(tmp);
        pl_str_builder_const_str(gen_params.shader_body, pass->compare_bodies[1]);
        pass->compare_tap_body_hash = pl_str_builder_hash(gen_params.shader_body);
        free(tap);
        a0_body = gen_params.shader_body;
        if (compare->uv_oracle) {
            char *oracle = compare->uv_oracle_probe
                ? mks_uv_oracle_probe_instrument(original, &site,
                    sh_ident_tostr(compare_words), compare->x, compare->y,
                    compare->width, compare->height, pass->compare_probe_binding, compare->repetitions)
                : mks_uv_oracle_instrument(original, &site,
                    sh_ident_tostr(compare_words), compare->x, compare->y,
                    compare->width, compare->height);
            if (!oracle) {
                PL_ERR(dp, "MKS-YUV-COMPARE-LIB event=uv-oracle-rejected reason=unsupported-site");
                goto error;
            }
            gen_params.shader_body = pl_str_builder_alloc(tmp);
            pl_str_builder_const_str(gen_params.shader_body, pl_strdup0(tmp, pl_str0(oracle)));
            free(oracle);
            pl_hash_merge(&pass->signature, UINT64_C(0x55564f5241434c45));
            if (compare->uv_oracle_single_pair)
                pl_hash_merge(&pass->signature, UINT64_C(0x53494e474c455556));
        }
    }
    if (gen_params.uv_table_storage && !gen_params.uv_table_ident) {
        // Production table rewrite-site discovery: find the single direct
        // sample tap in the finalized body (the mechanism schema10 validated
        // for the probe substitution). A body without exactly one supported
        // tap must not fuse untabled.
        pl_str body = pl_str_builder_exec(gen_params.shader_body);
        char *original = pl_strdup0(tmp, body);
        struct mks_sample_tap_site site;
        bool supported = dp->gpu->glsl.gles && dp->gpu->glsl.version >= 310 &&
            params.type == PL_PASS_RASTER && mks_sample_tap_find(original, &site);
        unsigned int matching_tex = 0, matching_uv = 0;
        pl_tex tap_tex = NULL;
        if (supported) {
            for (int i = 0; i < sh->descs.num; i++) {
                const struct pl_shader_desc *sd = &sh->descs.elem[i];
                if (sd->desc.type != PL_DESC_SAMPLED_TEX)
                    continue;
                pl_tex tex = sd->binding.object;
                if (sh_ident_unpack(sd->desc.name) == site.tex_id && tex &&
                    tex->sampler_type == PL_SAMPLER_NORMAL && !tex->params.d &&
                    tex->params.w > 0 && tex->params.h > 0) {
                    matching_tex++;
                    tap_tex = tex;
                }
            }
            for (int i = 0; i < sh->vas.num; i++) {
                const struct pl_vertex_attrib *va = &sh->vas.elem[i].attr;
                if (sh_ident_unpack(va->name) == site.uv_id && va->fmt &&
                    va->fmt->type == PL_FMT_FLOAT && va->fmt->num_components == 2)
                    matching_uv++;
            }
        }
        if (!supported || matching_tex != 1 || matching_uv != 1) {
            PL_ERR(dp->gpu, "MKS-FUSION-TABLE event=table-required-unmet reason=tap-not-unique");
            goto error;
        }
#ifdef PL_HAVE_OPENGL
        // Final-tap material identity: the table's UV entries are valid only
        // for the exact tap texture geometry recorded at seeding. A renderer
        // path change (runtime peak-detect flip, a still-frame single-frame
        // cache FBO) swaps the tap onto a different texture — same high-level
        // source config, different sampling geometry — and must not consume
        // this table. Compare the actual texture against the seed material.
        // (GL-only API; without OpenGL no table can be attached and this
        // block is unreachable — the guard keeps the no-GL link clean.)
        const struct pl_opengl_uv_atlas_material *seed_mat =
            pl_opengl_uv_atlas_table_seed_material(dp->gpu);
        if (!seed_mat ||
            tap_tex->params.w != seed_mat->input_width ||
            tap_tex->params.h != seed_mat->input_height ||
            tap_tex->sampler_type != seed_mat->sampler_type ||
            !tap_tex->params.format ||
            tap_tex->params.format->type != seed_mat->format_type ||
            tap_tex->params.format->num_components != seed_mat->num_components) {
            PL_ERR(dp->gpu, "MKS-FUSION-TABLE event=table-input-mismatch tap=%dx%d seed=%dx%d",
                   tap_tex->params.w, tap_tex->params.h,
                   seed_mat ? seed_mat->input_width : -1,
                   seed_mat ? seed_mat->input_height : -1);
            goto error;
        }
        // Corner-UV identity: the archived table maps output pixels to UV
        // endpoints; the seeding pass's four corner UV floats are the exact
        // mapping numerics. Byte-compare the current pass's corners against
        // the archive so any same-texture drift (rect change, endpoint
        // re-rounding through a different path) refuses fusion.
        {
            const float *seed_uv = pl_opengl_uv_atlas_table_seed_uv(dp->gpu);
            float corners[8];
            bool have_uv = false;
            for (int i = 0; i < sh->vas.num; i++) {
                if (sh_ident_unpack(sh->vas.elem[i].attr.name) != site.uv_id)
                    continue;
                const struct pl_shader_va *sva = &sh->vas.elem[i];
                for (int n = 0; n < 4; n++)
                    memcpy(&corners[n * 2], sva->data[n],
                           PL_MIN(sizeof(float) * 2, (size_t) sva->attr.fmt->texel_size));
                have_uv = true;
                break;
            }
            if (!seed_uv || !have_uv || memcmp(corners, seed_uv, sizeof(corners)) != 0) {
                PL_ERR(dp->gpu, "MKS-FUSION-TABLE event=table-uv-drift");
                goto error;
            }
        }
#endif
        gen_params.uv_table_ident = site.uv_id;
    }
    generate_shaders(dp, &gen_params, &vert_builder, &glsl_builder);
    char *compare_glsl = NULL, *a0_vertex = NULL;
    if (compare) {
        // Materialize primary source before reusing dispatch scratch builders.
        pl_str primary = pl_str_builder_exec(glsl_builder);
        char *saved_primary = pl_strdup0(tmp, primary);
        pl_str vertex = pl_str_builder_exec(vert_builder);
        char *saved_vertex = pl_strdup0(tmp, vertex);
        a0_vertex = saved_vertex;
        for (int n=0; n<TMP_COUNT; n++)
            pl_str_builder_reset(dp->tmp[n]);
        gen_params.rgb16f_to_yuv = false;
        if (compare->uv_oracle)
            gen_params.shader_body = a0_body;
        generate_shaders(dp, &gen_params, &vert_builder, &glsl_builder);
        pl_str companion = pl_str_builder_exec(glsl_builder);
        compare_glsl = pl_strdup0(tmp, companion);
        // Both generated programs are part of the one diagnostic cache key.
        for (int n=0; n<TMP_COUNT; n++)
            pl_str_builder_reset(dp->tmp[n]);
        if (compare->invariant_b) {
            char *modified = compare->sample_tap
                ? mks_invariant_b_vertex(saved_vertex, pass->compare_tap_varying) : NULL;
            if (!modified) {
                PL_ERR(dp, "MKS-YUV-COMPARE-LIB event=invariant-b-rejected reason=unsupported-vertex");
                goto error;
            }
            saved_vertex = pl_strdup0(tmp, pl_str0(modified));
            free(modified);
            // Isolate both the diagnostic mode and the exact transformed source.
            pl_hash_merge(&pass->signature, UINT64_C(0x494e564152424f4e));
            pl_hash_merge(&pass->signature, pl_str0_hash(saved_vertex));
        }
        pl_str_builder_const_str(dp->tmp[TMP_MAIN], saved_primary);
        pl_str_builder_const_str(dp->tmp[TMP_VERT_HEAD], saved_vertex);
        glsl_builder = dp->tmp[TMP_MAIN];
        vert_builder = dp->tmp[TMP_VERT_HEAD];
        pl_hash_merge(&pass->signature, compare->baseline->params.format->signature);
    }
    for (int i = 0; i < dp->passes.num; i++) {
        struct pass *p = dp->passes.elem[i];
        if (p->signature != pass->signature)
            continue;

        // Found existing shader, re-use directly. Diagnostic body provenance
        // belongs to this capture even if only source trivia was canonicalized.
        if (compare) {
            p->compare_body_hash = pass->compare_body_hash;
            p->compare_tap_body_hash = pass->compare_tap_body_hash;
            for (int n = 0; n < 2; n++) {
                pl_free(p->compare_bodies[n]);
                p->compare_bodies[n] = pl_steal(p, pass->compare_bodies[n]);
                pass->compare_bodies[n] = NULL;
            }
        }
        if (p->ubo)
            sh->descs.elem[p->ubo_index].binding.object = p->ubo;
        pl_free(p->run_params.constant_data);
        p->run_params.constant_data = pl_steal(p, constant_data);
        p->last_index = dp->current_index;
        pl_free(pass);
        return p;
    }

    // Need to compile new shader, execute templates now
    if (vert_builder) {
        pl_str vert = pl_str_builder_exec(vert_builder);
        params.vertex_shader = (char *) vert.buf;
    }
    pl_str glsl = pl_str_builder_exec(glsl_builder);
    params.glsl_shader = (char *) glsl.buf;

    // Turn all shader identifiers into actual strings before passing it
    // to the `pl_gpu`
#define FIX_IDENT(name) \
    name = sh_ident_tostr(sh_ident_unpack(name))
    for (int i = 0; i < params.num_variables; i++)
        FIX_IDENT(params.variables[i].name);
    for (int i = 0; i < params.num_descriptors; i++)
        FIX_IDENT(params.descriptors[i].name);
    for (int i = 0; i < params.num_vertex_attribs; i++)
        FIX_IDENT(params.vertex_attribs[i].name);
#undef FIX_IDENT

    // This uniform belongs only to B. It must not change A0 declarations,
    // variable placement, source, or program. Its value is not in the cache key.
    int a0_num_variables = params.num_variables;
    int a0_num_descriptors = params.num_descriptors;
    if (compare && compare->uv_oracle) {
        pass->compare_oracle_var = params.num_variables;
        PL_ARRAY_APPEND_RAW(tmp, params.variables, params.num_variables,
                            pl_var_int("mks_uv_oracle_use_a"));
    }
    if (compare && compare->uv_oracle_probe) {
        pass->compare_probe_pair_var = params.num_variables;
        PL_ARRAY_APPEND_RAW(tmp, params.variables, params.num_variables,
                            pl_var_int("mks_probe_pair"));
        pass->compare_probe_desc_index = params.num_descriptors;
        PL_ARRAY_APPEND_RAW(tmp, params.descriptors, params.num_descriptors,
            ((struct pl_desc) { .name = "mks_oracle_probe", .type = PL_DESC_BUF_STORAGE,
                .access = PL_DESC_ACCESS_READWRITE, .binding = pass->compare_probe_binding }));
    }
    pass->pass = pl_pass_create(dp->gpu, &params);
    if (compare) {
        struct pl_pass_params companion = params;
        companion.glsl_shader = compare_glsl;
        companion.target_format = compare->baseline->params.format;
        companion.vertex_shader = a0_vertex;
        companion.num_variables = a0_num_variables;
        companion.num_descriptors = a0_num_descriptors;
        pass->compare_baseline = pl_pass_create(dp->gpu, &companion);
        pass->compare_sources[0] = pl_strdup0(pass, pl_str0(params.glsl_shader));
        pass->compare_sources[1] = pl_strdup0(pass, pl_str0(companion.glsl_shader));
        // Preserve the common/A0 vertex for legacy source getters. The B-only
        // variant is archived separately and is never used to compile A0.
        pass->compare_sources[2] = pl_strdup0(pass, pl_str0(companion.vertex_shader));
        if (compare->invariant_b) {
            pass->compare_vertices[0] = pl_strdup0(pass, pl_str0(params.vertex_shader));
            pass->compare_vertices[1] = pl_strdup0(pass, pl_str0(companion.vertex_shader));
        }
    }
    if (!pass->pass) {
        PL_ERR(dp, "Failed creating render pass for dispatch");
        // Add it anyway
    }

    struct pl_pass_run_params *rparams = &pass->run_params;
    rparams->pass = pass->pass;
    rparams->constant_data = constant_data;
    rparams->push_constants = pl_zalloc(pass, params.push_constants_size);
    rparams->desc_bindings = pl_calloc_ptr(pass, params.num_descriptors,
                                           rparams->desc_bindings);

    if (ubo_size && (pass->pass || pass->compare_baseline)) {
        // Create the UBO
        pass->ubo = pl_buf_create(dp->gpu, pl_buf_params(
            .size = ubo_size,
            .uniform = true,
            .host_writable = true,
        ));

        if (!pass->ubo) {
            PL_ERR(dp, "Failed creating uniform buffer for dispatch");
            goto error;
        }

        sh->descs.elem[pass->ubo_index].binding.object = pass->ubo;
    }

    if (params.type == PL_PASS_RASTER && !vparams) {
        // Generate the vertex array placeholder
        rparams->vertex_count = 4; // single quad
        size_t vert_size = rparams->vertex_count * params.vertex_stride;
        rparams->vertex_data = pl_zalloc(pass, vert_size);
        pass->owned_vertex_data = rparams->vertex_data;
        pass->owned_vertex_bytes = vert_size;
    }

    pass->timer = pl_timer_create(dp->gpu);

    PL_ARRAY_APPEND(dp, dp->passes, pass);
    return pass;

error:
    pass_destroy(dp, pass);
    return NULL;
}

static void update_pass_var(pl_dispatch dp, struct pass *pass,
                            const struct pl_shader_var *sv, struct pass_var *pv)
{
    struct pl_var_layout host_layout = pl_var_host_layout(0, &sv->var);
    pl_assert(host_layout.size);

    // Use the cache to skip updates if possible
    if (pv->cached_data && !memcmp(sv->data, pv->cached_data, host_layout.size))
        return;
    if (!pv->cached_data)
        pv->cached_data = pl_alloc(pass, host_layout.size);
    memcpy(pv->cached_data, sv->data, host_layout.size);

    struct pl_pass_run_params *rparams = &pass->run_params;
    switch (pv->type) {
    case PASS_VAR_NONE:
        pl_unreachable();
    case PASS_VAR_GLOBAL: {
        struct pl_var_update vu = {
            .index = pv->index,
            .data  = sv->data,
        };
        PL_ARRAY_APPEND_RAW(pass, rparams->var_updates, rparams->num_var_updates, vu);
        break;
    }
    case PASS_VAR_UBO: {
        pl_assert(pass->ubo);
        const size_t offset = pv->layout.offset;
        if (host_layout.stride == pv->layout.stride) {
            pl_assert(host_layout.size == pv->layout.size);
            pl_buf_write(dp->gpu, pass->ubo, offset, sv->data, host_layout.size);
        } else {
            // Coalesce strided UBO write into a single pl_buf_write to avoid
            // unnecessary synchronization overhead by assembling the correctly
            // strided upload in RAM
            pl_grow(dp, &dp->ubo_tmp, pv->layout.size);
            uint8_t * const tmp = dp->ubo_tmp;
            const uint8_t *src = sv->data;
            const uint8_t *end = src + host_layout.size;
            uint8_t *dst = tmp;
            while (src < end) {
                memcpy(dst, src, host_layout.stride);
                src += host_layout.stride;
                dst += pv->layout.stride;
            }
            pl_buf_write(dp->gpu, pass->ubo, offset, tmp, pv->layout.size);
        }
        break;
    }
    case PASS_VAR_PUSHC:
        pl_assert(rparams->push_constants);
        memcpy_layout(rparams->push_constants, pv->layout, sv->data, host_layout);
        break;
    };
}

static void compute_vertex_attribs(pl_dispatch dp, pl_shader sh,
                                   int width, int height, ident_t *out_scale)
{
    // Simulate vertex attributes using global definitions
    *out_scale = sh_var(sh, (struct pl_shader_var) {
        .var     = pl_var_vec2("out_scale"),
        .data    = &(float[2]){ 1.0 / width, 1.0 / height },
        .dynamic = true,
    });

    GLSLP("#define frag_pos(id) (vec2(id) + vec2(0.5))  \n"
          "#define frag_map(id) ("$" * frag_pos(id))    \n"
          "#define gl_FragCoord vec4(frag_pos(gl_GlobalInvocationID), 0.0, 1.0) \n",
          *out_scale);

    for (int n = 0; n < sh->vas.num; n++) {
        const struct pl_shader_va *sva = &sh->vas.elem[n];

        ident_t points[4];
        for (int i = 0; i < PL_ARRAY_SIZE(points); i++) {
            points[i] = sh_var(sh, (struct pl_shader_var) {
                .var  = pl_var_from_fmt(sva->attr.fmt, "pt"),
                .data = sva->data[i],
            });
        }

        GLSLP("#define "$"_map(id) "
             "(mix(mix("$", "$", frag_map(id).x), "
             "     mix("$", "$", frag_map(id).x), "
             "frag_map(id).y)) \n"
             "#define "$" ("$"_map(gl_GlobalInvocationID)) \n",
             sh_ident_unpack(sva->attr.name),
             points[0], points[1], points[2], points[3],
             sh_ident_unpack(sva->attr.name),
             sh_ident_unpack(sva->attr.name));
    }
}

static const char *map_blend_mode(enum pl_blend_mode mode)
{
    switch (mode) {
    case PL_BLEND_ZERO:                return "0.0";
    case PL_BLEND_ONE:                 return "1.0";
    case PL_BLEND_SRC_ALPHA:           return "color.a";
    case PL_BLEND_ONE_MINUS_SRC_ALPHA: return "(1.0 - color.a)";
    case PL_BLEND_MODE_COUNT: break;
    }

    pl_unreachable();
}

static void translate_compute_shader(pl_dispatch dp, pl_shader sh,
                                     const pl_rect2d *rc,
                                     const struct pl_dispatch_params *params)
{
    int width = abs(pl_rect_w(*rc)), height = abs(pl_rect_h(*rc));
    if (sh->transpose)
        PL_SWAP(width, height);
    ident_t out_scale;
    compute_vertex_attribs(dp, sh, width, height, &out_scale);

    // Simulate a framebuffer using storage images
    pl_assert(params->target->params.storable);
    pl_assert(sh->output == PL_SHADER_SIG_COLOR);
    ident_t fbo = sh_desc(sh, (struct pl_shader_desc) {
        .binding.object = params->target,
        .desc = {
            .name    = "out_image",
            .type    = PL_DESC_STORAGE_IMG,
            .access  = params->blend_params ? PL_DESC_ACCESS_READWRITE
                                            : PL_DESC_ACCESS_WRITEONLY,
        },
    });

    ident_t base = sh_var(sh, (struct pl_shader_var) {
        .data    = &(int[2]){
            // When dealing with flipped coordinates, we need to subtract one
            // from the effective sample position; this is because instead of
            // IDs normally spanning from [0, size - 1], the inverted math
            // would end up with [size, 1] instead.
            //
            // Put another way, because we are dealing with integer coordinates,
            // the calculated position is rounded towards zero - but when the
            // coordinate system is flipped, that ends up rounding up instead
            // of down.
            rc->x0 - (rc->x0 > rc->x1),
            rc->y0 - (rc->y0 > rc->y1),
        },
        .dynamic = true,
        .var     = {
            .name  = "base",
            .type  = PL_VAR_SINT,
            .dim_v = 2,
            .dim_m = 1,
            .dim_a = 1,
        },
    });

    int dx = rc->x0 > rc->x1 ? -1 : 1, dy = rc->y0 > rc->y1 ? -1 : 1;
    GLSL("ivec2 dir = ivec2(%d, %d);\n", dx, dy); // hard-code, not worth var
    GLSL("ivec2 pos = "$" + dir * ivec2(gl_GlobalInvocationID).%c%c;\n",
         base, sh->transpose ? 'y' : 'x', sh->transpose ? 'x' : 'y');
    GLSL("vec2 fpos = "$" * vec2(gl_GlobalInvocationID);\n", out_scale);
    GLSL("if (fpos.x < 1.0 && fpos.y < 1.0) {\n");
    if (params->blend_params) {
        GLSL("vec4 orig = imageLoad("$", pos);\n", fbo);
        GLSL("color = vec4(color.rgb * vec3(%s), color.a * %s) \n"
             "      + vec4(orig.rgb  * vec3(%s), orig.a  * %s);\n",
             map_blend_mode(params->blend_params->src_rgb),
             map_blend_mode(params->blend_params->src_alpha),
             map_blend_mode(params->blend_params->dst_rgb),
             map_blend_mode(params->blend_params->dst_alpha));
    }
    GLSL("imageStore("$", pos, color);\n", fbo);
    GLSL("}\n");
    sh->output = PL_SHADER_SIG_NONE;
}

static void run_pass(pl_dispatch dp, pl_shader sh, struct pass *pass)
{
    pl_shader_info shader = &sh->info->info;
    pl_pass_run(dp->gpu, &pass->run_params);

    for (uint64_t ts; (ts = pl_timer_query(dp->gpu, pass->timer));) {
        PL_TRACE(dp, "Spent %.3f ms on shader: %s", ts / 1e6, shader->description);

        uint64_t old = pass->samples[pass->ts_idx];
        pass->samples[pass->ts_idx] = ts;
        pass->ts_last = ts;
        pass->ts_peak = PL_MAX(pass->ts_peak, ts);
        pass->ts_sum += ts;
        pass->ts_idx = (pass->ts_idx + 1) % PL_ARRAY_SIZE(pass->samples);

        if (old) {
            pass->ts_sum -= old;
            if (old == pass->ts_peak) {
                uint64_t new_peak = 0;
                for (int i = 0; i < PL_ARRAY_SIZE(pass->samples); i++)
                    new_peak = PL_MAX(new_peak, pass->samples[i]);
                pass->ts_peak = new_peak;
            }
        }
    }

    if (!dp->info_callback)
        return;

    struct pl_dispatch_info info;
    info.signature = pass->signature;
    info.shader = shader;

    // Test to see if the ring buffer already wrapped around once
    if (pass->samples[pass->ts_idx]) {
        info.num_samples = PL_ARRAY_SIZE(pass->samples);
        int num_wrapped = info.num_samples - pass->ts_idx;
        memcpy(info.samples, &pass->samples[pass->ts_idx],
               num_wrapped * sizeof(info.samples[0]));
        memcpy(&info.samples[num_wrapped], pass->samples,
               pass->ts_idx * sizeof(info.samples[0]));
    } else {
        info.num_samples = pass->ts_idx;
        memcpy(info.samples, pass->samples,
               pass->ts_idx * sizeof(info.samples[0]));
    }

    info.last = pass->ts_last;
    info.peak = pass->ts_peak;
    info.average = pass->ts_sum / PL_MAX(info.num_samples, 1);
    dp->info_callback(dp->info_priv, &info);
}

// Strict bounded host-quad diagnostic. This never mutates the cached A data.
static bool compare_quad_text(char *out, size_t size, const char *value)
{
    if (!value || !*value || strlen(value) >= size)
        return false;
    for (const char *p = value; *p; p++) {
        if (!(isalnum((unsigned char) *p) || *p == '_'))
            return false;
    }
    memcpy(out, value, strlen(value) + 1);
    return true;
}

static bool compare_quad_copy(pl_shader sh, const struct pass *pass,
    const struct pl_pass_run_params *a, const struct pl_pass_run_params *b,
    bool reverse, struct pl_opengl_yuv_compare_quad_layout *layout,
    const uint8_t **candidate)
{
    if (!a->pass || !b->pass || !a->target || !b->target ||
        b->vertex_count != 4 || a->vertex_count != 4 || !b->vertex_data ||
        b->vertex_data != pass->owned_vertex_data || a->vertex_data != b->vertex_data ||
        b->vertex_buf || a->vertex_buf || b->index_data || a->index_data ||
        b->index_buf || a->index_buf || b->buf_offset || a->buf_offset ||
        b->index_offset || a->index_offset)
        return false;
    const struct pl_pass_params *bp = &b->pass->params, *ap = &a->pass->params;
    if (bp->type != PL_PASS_RASTER || ap->type != PL_PASS_RASTER ||
        bp->vertex_type != PL_PRIM_TRIANGLE_STRIP || ap->vertex_type != bp->vertex_type ||
        bp->blend_params || ap->blend_params || !bp->vertex_shader || !ap->vertex_shader ||
        strcmp(bp->vertex_shader, ap->vertex_shader) ||
        !bp->vertex_stride || bp->vertex_stride > 1024 || bp->vertex_stride % 4 ||
        ap->vertex_stride != bp->vertex_stride || pass->owned_vertex_bytes != bp->vertex_stride * 4 ||
        bp->num_vertex_attribs < 1 || bp->num_vertex_attribs > PL_OPENGL_YUV_COMPARE_QUAD_MAX_ATTRIBS ||
        ap->num_vertex_attribs != bp->num_vertex_attribs || !bp->vertex_attribs || !ap->vertex_attribs ||
        pass->position_attrib < 0 || pass->position_attrib >= bp->num_vertex_attribs ||
        a->target->params.w != b->target->params.w || a->target->params.h != b->target->params.h ||
        b->target->params.w <= 0 || b->target->params.h <= 0 ||
        memcmp(&a->viewport, &b->viewport, sizeof(a->viewport)) ||
        memcmp(&a->scissors, &b->scissors, sizeof(a->scissors)))
        return false;
    *layout = (struct pl_opengl_yuv_compare_quad_layout) {
        .primitive = bp->vertex_type, .indexed = false, .vertex_count = b->vertex_count,
        .num_attributes = bp->num_vertex_attribs, .position_attrib = pass->position_attrib,
        .stride = bp->vertex_stride, .byte_size = pass->owned_vertex_bytes,
        .viewport = b->viewport, .scissors = b->scissors,
    };
    // Mirror only the ordinary pl_pass_run normalization; submitted run params
    // remain untouched. Restrict this experiment to the actual full viewport.
    if (!layout->viewport.x0 && !layout->viewport.x1) layout->viewport.x1 = b->target->params.w;
    if (!layout->viewport.y0 && !layout->viewport.y1) layout->viewport.y1 = b->target->params.h;
    if (!layout->scissors.x0 && !layout->scissors.x1) layout->scissors.x1 = b->target->params.w;
    if (!layout->scissors.y0 && !layout->scissors.y1) layout->scissors.y1 = b->target->params.h;
    layout->scissors.x0 = PL_CLAMP(layout->scissors.x0, 0, b->target->params.w);
    layout->scissors.x1 = PL_CLAMP(layout->scissors.x1, 0, b->target->params.w);
    layout->scissors.y0 = PL_CLAMP(layout->scissors.y0, 0, b->target->params.h);
    layout->scissors.y1 = PL_CLAMP(layout->scissors.y1, 0, b->target->params.h);
    pl_rect2d full = {0, 0, b->target->params.w, b->target->params.h};
    if (!pl_rect2d_eq(layout->viewport, full) || !pl_rect2d_eq(layout->scissors, full))
        return false;
    static const char *types[] = {NULL, "float", "vec2", "vec3", "vec4"};
    for (int i = 0; i < bp->num_vertex_attribs; i++) {
        const struct pl_vertex_attrib *v = &bp->vertex_attribs[i], *av = &ap->vertex_attribs[i];
        pl_fmt f = v->fmt;
        struct pl_opengl_yuv_compare_quad_attrib *o = &layout->attributes[i];
        if (!f || f != av->fmt || !(f->caps & PL_FMT_CAP_VERTEX) || f->opaque || f->num_planes ||
            f->type != PL_FMT_FLOAT || f->num_components < 1 || f->num_components > 4 ||
            f->texel_size != (size_t) f->num_components * 4 || v->offset % 4 ||
            v->offset > bp->vertex_stride || f->texel_size > bp->vertex_stride - v->offset ||
            v->location != i || v->location != av->location || v->offset != av->offset ||
            !v->name || !av->name || strcmp(v->name, av->name) ||
            !f->glsl_type || strcmp(f->glsl_type, types[f->num_components]) ||
            !compare_quad_text(o->name, sizeof(o->name), v->name) ||
            !compare_quad_text(o->format_name, sizeof(o->format_name), f->name) ||
            !compare_quad_text(o->glsl_type, sizeof(o->glsl_type), f->glsl_type))
            return false;
        for (int j = 0; j < i; j++) {
            const struct pl_vertex_attrib *old = &bp->vertex_attribs[j];
            if (old->location == v->location || !strcmp(old->name, v->name) ||
                (old->offset < v->offset + f->texel_size && v->offset < old->offset + old->fmt->texel_size))
                return false;
        }
        for (int k = 0; k < f->num_components; k++) {
            if (f->component_depth[k] != 32 || f->host_bits[k] != 32 || f->sample_order[k] != k)
                return false;
            for (int n = 0; n < 4; n++) {
                uint32_t raw;
                memcpy(&raw, (const uint8_t *) b->vertex_data + n * bp->vertex_stride + v->offset + k * 4, 4);
                if ((raw & 0x7f800000u) == 0x7f800000u)
                    return false;
            }
        }
        o->fmt_type = f->type;
        o->location = v->location; o->num_components = f->num_components;
        o->offset = v->offset; o->texel_size = f->texel_size;
        o->opaque = f->opaque; o->num_planes = f->num_planes;
        memcpy(o->component_depth, f->component_depth, sizeof(o->component_depth));
        memcpy(o->host_bits, f->host_bits, sizeof(o->host_bits));
        memcpy(o->sample_order, f->sample_order, sizeof(o->sample_order));
    }
    const struct pl_opengl_yuv_compare_quad_attrib *position = &layout->attributes[pass->position_attrib];
    if (position->num_components != 2)
        return false;
    char init[128];
    snprintf(init, sizeof(init), "vec2 va_pos = %s;", position->name);
    const char *vertex = bp->vertex_shader;
    const char *first = strstr(vertex, "va_pos");
    const char *second = first ? strstr(first + 6, "va_pos") : NULL;
    if (!strstr(vertex, init) || !strstr(vertex, "gl_Position = vec4(va_pos, 0.0, 1.0);") ||
        !second || strstr(second + 6, "va_pos"))
        return false; // no projection/offset/unknown position path
    float xy[4][2];
    unsigned int corners = 0;
    for (int n = 0; n < 4; n++) {
        memcpy(xy[n], (const uint8_t *) b->vertex_data + n * layout->stride + position->offset, sizeof(xy[n]));
        if ((xy[n][0] != -1.0f && xy[n][0] != 1.0f) || (xy[n][1] != -1.0f && xy[n][1] != 1.0f))
            return false;
        unsigned int bit = 1u << ((xy[n][0] > 0.0f) + 2 * (xy[n][1] > 0.0f));
        if (corners & bit) return false;
        corners |= bit;
    }
    if (corners != 15u || xy[1][0] != -xy[2][0] || xy[1][1] != -xy[2][1])
        return false;
    uint8_t *copy = pl_alloc(sh->tmp, layout->byte_size);
    if (!copy)
        return false;
    for (int n = 0; n < 4; n++)
        memcpy(copy + n * layout->stride, (const uint8_t *) b->vertex_data + (reverse ? 3 - n : n) * layout->stride,
               layout->stride);
    *candidate = copy;
    return true;
}

// Reuse the same frozen layout, inputs, geometry, and compiled programs. The
// only per-pair binding change is the diagnostic SSBO sink. No renderer reentry.
static bool run_compare_passes(pl_dispatch dp, pl_shader sh, struct pass *pass,
                                const struct pl_yuv_compare *compare, pl_tex target)
{
    struct pl_pass_run_params *b = &pass->run_params;
    unsigned int repetitions = compare->repetitions;
    int slot = pass->compare_desc_index;
    if ((compare->quad_order && (!compare->uv_oracle_single_pair ||
         compare->uv_oracle_use_a || !compare->quad_payloads)) ||
        (compare->quad_order_reverse && !compare->quad_order) ||
        (compare->uv_oracle_probe && (!compare->uv_oracle || !compare->probe_storage ||
         pass->compare_probe_desc_index < sh->descs.num || pass->compare_probe_pair_var < 0 ||
         pass->compare_probe_binding < 0 ||
         compare->probe_storage == compare->storage ||
         compare->probe_storage == compare->storage_repeat ||
         compare->probe_storage->params.size != (size_t) compare->width * compare->height * 10 * repetitions * sizeof(uint32_t))) ||
        (compare->uv_oracle_single_pair && (!compare->uv_oracle_probe ||
         !compare->uv_oracle || repetitions != 1)) ||
        !compare->record || repetitions < 1 || repetitions > 2 || slot < 0 ||
        (compare->sample_tap && ((repetitions != 2 && !compare->uv_oracle_single_pair) || compare->words_per_pixel != 26 ||
         !compare->bodies || !pass->compare_bodies[0] || !pass->compare_bodies[1])) ||
        (compare->uv_oracle && (!compare->sample_tap || compare->invariant_b ||
         !compare->barrier || pass->compare_oracle_var < 0)) ||
        (compare->invariant_b && (!compare->sample_tap || !compare->vertices ||
         !pass->compare_vertices[0] || !pass->compare_vertices[1] ||
         !pass->compare_tap_varying[0])) ||
        slot >= sh->descs.num || !compare->storage ||
        (repetitions == 2 && (!compare->storage_repeat ||
         compare->storage_repeat == compare->storage ||
         compare->storage_repeat->params.size != compare->storage->params.size)))
    {
        compare->finish(dp->gpu, target, false, false, pass->compare_body_hash);
        return false;
    }
    if (compare->sample_tap)
        compare->bodies(dp->gpu, target, pass->compare_bodies[0], pass->compare_bodies[1],
                        pass->compare_tap_body_hash);
    compare->sources(dp->gpu, target, pass->compare_sources[0],
                     pass->compare_sources[1], pass->compare_sources[2]);
    if (compare->invariant_b)
        compare->vertices(dp->gpu, target, pass->compare_vertices[0], pass->compare_vertices[1],
                          pass->compare_tap_varying);
    // A is a distinct program, so supply every GLOBAL value, not just B's
    // incremental updates. This exact immutable set is reused by A1 and A2.
    struct pl_pass_run_params a = *b;
    a.pass = pass->compare_baseline;
    a.target = compare->baseline;
    a.timer = b->timer = NULL;
    struct pl_var_update *updates = pl_calloc_ptr(sh->tmp, sh->vars.num, updates);
    a.var_updates = updates;
    a.num_var_updates = 0;
    for (int i = 0; i < sh->vars.num; i++) {
        if (pass->vars[i].type == PASS_VAR_GLOBAL)
            updates[a.num_var_updates++] = (struct pl_var_update) {
                .index = pass->vars[i].index, .data = sh->vars.elem[i].data,
            };
    }
    struct pl_pass_run_params oracle_b;
    int oracle_mode = compare->uv_oracle_use_a ? 1 : 0;
    int probe_pair = 0;
    if (compare->uv_oracle) {
        oracle_b = *b;
        struct pl_var_update *bu = pl_calloc_ptr(sh->tmp, b->num_var_updates + 1 + compare->uv_oracle_probe, bu);
        memcpy(bu, b->var_updates, b->num_var_updates * sizeof(*bu));
        bu[b->num_var_updates] = (struct pl_var_update) {
            .index = pass->compare_oracle_var, .data = &oracle_mode,
        };
        oracle_b.var_updates = bu;
        oracle_b.num_var_updates++;
        if (compare->uv_oracle_probe) {
            bu[oracle_b.num_var_updates++] = (struct pl_var_update) {
                .index = pass->compare_probe_pair_var, .data = &probe_pair,
            };
            b->desc_bindings[pass->compare_probe_desc_index].object = compare->probe_storage;
        }
    }
    if (compare->quad_order) {
        struct pl_opengl_yuv_compare_quad_layout layout;
        const uint8_t *quad_b = NULL;
        if (!compare_quad_copy(sh, pass, &a, b, compare->quad_order_reverse, &layout, &quad_b) ||
            !compare->quad_payloads(dp->gpu, target, b->vertex_data, quad_b, &layout))
        {
            compare->finish(dp->gpu, target, false, false, pass->compare_body_hash);
            return false;
        }
        oracle_b.vertex_data = quad_b;
    }
    bool all_b = true, all_a = true, last_a = false;
    for (unsigned int n = 0; n < repetitions; n++) {
        probe_pair = n;
        b->desc_bindings[slot].object = n ? compare->storage_repeat : compare->storage;
        for (unsigned int kind = 0; kind < 2; kind++) {
            bool is_a = compare->uv_oracle ? !kind : kind;
            struct pl_pass_run_params *run = is_a ? &a :
                (compare->uv_oracle ? &oracle_b : b);
            uint64_t before = 0, after = 0;
            bool ok = run->pass && compare->prepare(dp->gpu, target, pass->compare_binding) &&
                      compare->checkpoint(dp->gpu, &before);
            if (ok && !is_a && compare->uv_oracle_probe)
                ok = compare->prepare(dp->gpu, target, pass->compare_probe_binding);
            if (ok) {
                pl_pass_run(dp->gpu, run);
                ok = compare->checkpoint(dp->gpu, &after) && before == after;
            }
            compare->record(dp->gpu, target, n * 2 + kind, run->pass, ok, before, after);
            if (is_a) {
                if (compare->uv_oracle)
                    ok = ok && compare->barrier(dp->gpu, target, n);
                all_a &= ok;
                last_a = ok;
            } else {
                all_b &= ok;
            }
        }
    }
    b->desc_bindings[slot].object = compare->storage;
    // A failed earlier draw prevents a complete numeric transaction. The last
    // successful A may still safely supply the separate original RGB receipt.
    compare->finish(dp->gpu, target, all_b, all_a, pass->compare_body_hash);
    return last_a;
}

// One renderer entry and one true ordinary shader finalization. The cached
// normal A0 pass/source/ABI is never mutated to build the atlas companion.
static bool atlas_material(pl_shader sh, const struct pass *pass,
    struct pl_opengl_uv_atlas_material *m, const void **texture)
{
    memset(m,0,sizeof(*m));
    const struct pl_pass_params *p=&pass->pass->params;
    if (p->num_descriptors!=1 || p->descriptors[0].type!=PL_DESC_SAMPLED_TEX ||
        p->push_constants_size || p->num_constants ||
        sh->vars.num>PL_OPENGL_UV_ATLAS_MATERIAL_VARS) return false;
    pl_tex tex=pass->run_params.desc_bindings[0].object;
    if (!tex || !tex->params.format || tex->sampler_type!=PL_SAMPLER_NORMAL ||
        tex->params.d || tex->params.w<1 || tex->params.h<1) return false;
    pl_fmt f=tex->params.format;
    if (!f->name || strlen(f->name)>=sizeof(m->format_name) || f->opaque || f->num_planes) return false;
    m->input_width=tex->params.w; m->input_height=tex->params.h;
    m->sampler_type=tex->sampler_type;
    m->sample_mode=pass->run_params.desc_bindings[0].sample_mode;
    m->address_mode=pass->run_params.desc_bindings[0].address_mode;
    m->format_type=f->type; m->num_components=f->num_components;
    strcpy(m->format_name,f->name);
    memcpy(m->component_depth,f->component_depth,sizeof(m->component_depth));
    memcpy(m->host_bits,f->host_bits,sizeof(m->host_bits));
    memcpy(m->sample_order,f->sample_order,sizeof(m->sample_order));
    m->num_variables=sh->vars.num;
    for (int i=0;i<sh->vars.num;i++) {
        const struct pass_var *pv=&pass->vars[i];
        const struct pl_shader_var *sv=&sh->vars.elem[i];
        if (pv->type!=PASS_VAR_GLOBAL || pv->index<0 || pv->index>=p->num_variables) return false;
        const char *name=p->variables[pv->index].name;
        size_t bytes=pl_var_host_layout(0,&sv->var).size;
        if (!name || strlen(name)>=sizeof(m->uniform_names[i]) || !sv->data ||
            !bytes || bytes>sizeof(m->uniforms)-m->uniform_bytes) return false;
        strcpy(m->uniform_names[i],name);
        m->uniform_offsets[i]=m->uniform_bytes; m->uniform_sizes[i]=bytes;
        m->uniform_types[i]=sv->var.type;m->uniform_dim_v[i]=sv->var.dim_v;
        m->uniform_dim_m[i]=sv->var.dim_m;m->uniform_dim_a[i]=sv->var.dim_a;
        memcpy(m->uniforms+m->uniform_bytes,sv->data,bytes); m->uniform_bytes+=bytes;
    }
    *texture=tex; return true;
}

static bool run_atlas_reader(pl_dispatch dp, pl_shader sh, struct pass *pass,
    struct pl_uv_atlas_anchor *atlas, pl_tex target)
{
    bool ok=false;
    const struct pl_pass_params *original=&pass->pass->params;
    struct pl_opengl_uv_atlas_material material; const void *texture=NULL;
    struct pl_pass_run_params a=pass->run_params;
    a.timer=NULL;
    struct pl_opengl_yuv_compare_quad_layout layout; const uint8_t *identity=NULL;
    // The original output is independent of atlas validity. A clean key
    // rejection can still display this fresh A; GL/owner failures cannot.
    uint64_t before=0,after=0;
    ok=atlas->checkpoint(dp->gpu,target,&before);
    if (ok) {pl_pass_run(dp->gpu,&a);ok=atlas->checkpoint(dp->gpu,target,&after)&&before==after;}
    atlas->record(dp->gpu,target,0,a.pass,ok,before,after);
    if (!ok || !atlas->readback(dp->gpu,target,0)) {ok=false;goto done;}
    ok=false;
    if (!atlas->yuv_target || !atlas->probe_storage || !atlas->probe_readback ||
        !atlas_material(sh,pass,&material,&texture) ||
        !atlas->material(dp->gpu,target,&material) ||
        !atlas->input_identity(dp->gpu,target,texture) ||
        !compare_quad_copy(sh,pass,&a,&a,false,&layout,&identity)) goto done;
    char *b0=mks_atlas_b0(original->glsl_shader,atlas->shader_body,atlas->width,atlas->height);
    char *probe=b0 ? mks_atlas_probe(b0,atlas->width,atlas->height) : NULL;
    if (!b0 || !probe) {free(b0);free(probe);goto done;}
    struct pl_pass_params p=*original;
    p.glsl_shader=b0;
    p.descriptors=pl_alloc(sh->tmp,3*sizeof(*p.descriptors));
    p.descriptors[0]=original->descriptors[0];
    p.descriptors[1]=(struct pl_desc){.name="mks_cache_block",.type=PL_DESC_BUF_STORAGE,
        .access=PL_DESC_ACCESS_READONLY,.binding=0};
    p.num_descriptors=2;
    atlas->b0_pass=pl_pass_create(dp->gpu,&p);
    p.glsl_shader=probe; p.num_descriptors=3;
    p.descriptors[2]=(struct pl_desc){.name="mks_read_block",.type=PL_DESC_BUF_STORAGE,
        .access=PL_DESC_ACCESS_READWRITE,.binding=1};
    p.variables=pl_alloc(sh->tmp,(p.num_variables+1)*sizeof(*p.variables));
    memcpy(p.variables,original->variables,p.num_variables*sizeof(*p.variables));
    p.variables[p.num_variables++]=pl_var_int("mks_read_mode");
    atlas->companion=atlas->b0_pass ? pl_pass_create(dp->gpu,&p) : NULL;
    free(b0);free(probe);
    if (!atlas->b0_pass || !atlas->companion ||
        !atlas->begin(dp->gpu,target,original->glsl_shader,atlas->companion->params.glsl_shader,
                      original->vertex_shader,&layout,a.vertex_data)) {
        if (!atlas->b0_pass || !atlas->companion) atlas->reader_reject=4;
        goto done;
    }
    struct pl_var_update *updates=pl_calloc_ptr(sh->tmp,sh->vars.num+1,updates);
    a.var_updates=updates; a.num_var_updates=0; a.timer=NULL;
    for (int i=0;i<sh->vars.num;i++)
        updates[a.num_var_updates++]=(struct pl_var_update){.index=pass->vars[i].index,.data=sh->vars.elem[i].data};
    struct pl_pass_run_params b=a;
    b.pass=atlas->companion; b.target=atlas->yuv_target;
    b.desc_bindings=pl_calloc_ptr(sh->tmp,3,b.desc_bindings);
    b.desc_bindings[0]=a.desc_bindings[0];
    b.desc_bindings[1].object=atlas->storage;
    b.desc_bindings[2].object=atlas->probe_storage;
    int mode=0;
    updates[a.num_var_updates]=(struct pl_var_update){.index=p.num_variables-1,.data=&mode};
    b.num_var_updates++;
    ok=true;
    for (unsigned stage=1;ok && stage<3;stage++) {
        struct pl_pass_run_params *run=&b;
        before=after=0; mode=(int)stage-1;
        ok=atlas->checkpoint(dp->gpu,target,&before);
        if (ok) {pl_pass_run(dp->gpu,run);ok=atlas->checkpoint(dp->gpu,target,&after)&&before==after;}
        atlas->record(dp->gpu,target,stage,run->pass,ok,before,after);
        if (ok) ok=atlas->probe_readback(dp->gpu,target,mode);
    }
done:
    atlas->finish(dp->gpu,target,ok);
    // Whether the independent A receipt is usable is a backend decision;
    // returning false must not cause a second renderer invocation while armed.
    return ok;
}

static bool run_atlas_anchor(pl_dispatch dp, pl_shader sh, struct pass *pass,
                              struct pl_uv_atlas_anchor *atlas, pl_tex target)
{
    if (atlas->reader) return run_atlas_reader(dp,sh,pass,atlas,target);
    const char *gate = NULL;
    uint64_t initial_epoch;
    bool ok = atlas->checkpoint(dp->gpu, target, &initial_epoch) && pass->pass &&
              atlas->shader_body && atlas->storage && !atlas->companion;
    if (!ok) { gate = "prereq"; goto done; }
    const struct pl_pass_params *original = &pass->pass->params;
    if (original->type != PL_PASS_RASTER || original->blend_params ||
        original->vertex_type != PL_PRIM_TRIANGLE_STRIP)
    { ok = false; gate = "raster"; goto done; }
    struct mks_sample_tap_site site;
    if (!mks_sample_tap_find(atlas->shader_body, &site)) { ok = false; gate = "tap"; goto done; }
    unsigned int textures = 0, matching = 0, uv = 0;
    for (int i = 0; i < original->num_descriptors; i++) {
        const struct pl_desc *desc = &original->descriptors[i];
        if (desc->type != PL_DESC_SAMPLED_TEX && desc->type != PL_DESC_BUF_UNIFORM)
        { ok = false; gate = "desc-type"; goto done; }
        if (desc->type != PL_DESC_SAMPLED_TEX) continue;
        textures++;
        pl_tex tex = pass->run_params.desc_bindings[i].object;
        if (desc->name && !strcmp(desc->name, sh_ident_tostr(site.tex_id)) && tex &&
            tex->sampler_type == PL_SAMPLER_NORMAL && !tex->params.d &&
            tex->params.w > 0 && tex->params.h > 0)
            matching++;
    }
    for (int i = 0; i < sh->vas.num; i++) {
        if (sh_ident_unpack(sh->vas.elem[i].attr.name) == site.uv_id &&
            sh->vas.elem[i].attr.fmt->type == PL_FMT_FLOAT &&
            sh->vas.elem[i].attr.fmt->num_components == 2)
            uv++;
    }
    if (textures != 1 || matching != 1 || uv != 1) {
        PL_ERR(dp->gpu, "MKS-ANCHOR-DIAG gate=desc-count textures=%u matching=%u uv=%u",
               textures, matching, uv);
        ok = false; goto done;
    }
    struct pl_opengl_uv_atlas_material material; const void *texture=NULL;
    if (atlas->material && atlas_material(sh,pass,&material,&texture)) {
        if (!atlas->material(dp->gpu,target,&material) ||
            !atlas->input_identity(dp->gpu,target,texture)) {ok=false;gate="material";goto done;}
    }
    char *source = mks_atlas_fragment(original->glsl_shader, atlas->shader_body,
                                      atlas->width, atlas->height);
    if (!source) { ok = false; gate = "source"; goto done; }
    struct pl_pass_params companion = *original;
    companion.glsl_shader = source;
    companion.constant_data = pass->run_params.constant_data;
    companion.descriptors = pl_alloc(sh->tmp, (original->num_descriptors + 1) * sizeof(*companion.descriptors));
    memcpy(companion.descriptors, original->descriptors, original->num_descriptors * sizeof(*companion.descriptors));
    companion.descriptors[companion.num_descriptors++] = (struct pl_desc) {
        .name = "mks_uv_atlas_block", .type = PL_DESC_BUF_STORAGE,
        .access = PL_DESC_ACCESS_READWRITE, .binding = 0,
    };
    atlas->companion = pl_pass_create(dp->gpu, &companion);
    free(source);
    if (!atlas->companion) { ok = false; gate = "pass-create"; goto done; }
    if (!atlas->checkpoint(dp->gpu, target, &initial_epoch))
    { ok = false; gate = "checkpoint2"; goto done; }
    struct pl_pass_run_params a0 = pass->run_params;
    a0.timer = NULL;
    struct pl_var_update *updates = pl_calloc_ptr(sh->tmp, sh->vars.num, updates);
    a0.var_updates = updates; a0.num_var_updates = 0;
    for (int i = 0; i < sh->vars.num; i++) {
        if (pass->vars[i].type == PASS_VAR_GLOBAL)
            updates[a0.num_var_updates++] = (struct pl_var_update) {
                .index = pass->vars[i].index, .data = sh->vars.elem[i].data,
            };
    }
    struct pl_pass_run_params capture = a0;
    capture.pass = atlas->companion;
    capture.desc_bindings = pl_alloc(sh->tmp, companion.num_descriptors * sizeof(*capture.desc_bindings));
    memcpy(capture.desc_bindings, a0.desc_bindings, original->num_descriptors * sizeof(*capture.desc_bindings));
    capture.desc_bindings[original->num_descriptors] = (struct pl_desc_binding) { .object = atlas->storage };
    struct pl_opengl_yuv_compare_quad_layout layout;
    const uint8_t *identity_copy = NULL;
    if (!compare_quad_copy(sh, pass, &a0, &capture, false, &layout, &identity_copy))
    { ok = false; gate = "quad-copy"; goto done; }
    if (!atlas->begin(dp->gpu, target, original->glsl_shader, atlas->companion->params.glsl_shader,
                      original->vertex_shader, &layout, a0.vertex_data))
    { ok = false; gate = "begin"; goto done; }
#ifdef PL_HAVE_OPENGL
    if (atlas->uv_corners) {
        // Archive the seeding pass's four corner UV values: the actual
        // mapping numerics the recorded table is valid for. The production
        // consumer gate byte-compares these on every table-driven pass.
        float corners[8];
        bool have_uv = false;
        for (int i = 0; i < sh->vas.num; i++) {
            if (sh_ident_unpack(sh->vas.elem[i].attr.name) != site.uv_id)
                continue;
            const struct pl_shader_va *sva = &sh->vas.elem[i];
            for (int n = 0; n < 4; n++) {
                memcpy(&corners[n * 2], sva->data[n],
                       PL_MIN(sizeof(float) * 2, (size_t) sva->attr.fmt->texel_size));
            }
            have_uv = true;
            break;
        }
        if (!have_uv || !atlas->uv_corners(dp->gpu, target, corners))
        { ok = false; gate = "uv-corners"; goto done; }
    }
#endif
    ok = true;
    for (unsigned int stage = 0; ok && stage < 3; stage++) {
        struct pl_pass_run_params *run = stage == 1 ? &capture : &a0;
        uint64_t before = 0, after = 0;
        ok = atlas->checkpoint(dp->gpu, target, &before);
        if (!ok) { gate = "stage-checkpoint"; break; }
        pl_pass_run(dp->gpu, run);
        ok = atlas->checkpoint(dp->gpu, target, &after) && before == after;
        if (!ok) { gate = "stage-postrun"; break; }
        atlas->record(dp->gpu, target, stage, run->pass, ok, before, after);
        ok = atlas->readback(dp->gpu, target, stage);
        if (!ok) { gate = "stage-readback"; break; }
    }
done:
    if (!ok && gate)
        PL_ERR(dp->gpu, "MKS-ANCHOR-DIAG gate=%s", gate);
    atlas->finish(dp->gpu, target, ok);
    return ok;
}

bool pl_dispatch_finish(pl_dispatch dp, const struct pl_dispatch_params *params)
{
    struct pl_uv_atlas_anchor *atlas = NULL;
    pl_shader sh = *params->shader;
    bool ret = false;
    pl_mutex_lock(&dp->lock);

    if (sh->failed) {
        PL_ERR(sh, "Trying to dispatch a failed shader.");
        goto error;
    }

    if (!sh->mutable) {
        PL_ERR(dp, "Trying to dispatch non-mutable shader?");
        goto error;
    }

    if (sh->input != PL_SHADER_SIG_NONE || sh->output != PL_SHADER_SIG_COLOR) {
        PL_ERR(dp, "Trying to dispatch shader with incompatible signature!");
        goto error;
    }

    const struct pl_tex_params *tpars = &params->target->params;
    if (pl_tex_params_dimension(*tpars) != 2 || !tpars->renderable) {
        PL_ERR(dp, "Trying to dispatch a shader using an invalid target "
               "texture. The target must be a renderable 2D texture.");
        goto error;
    }

    const struct pl_gpu_limits *limits = &dp->gpu->limits;
    bool can_compute = tpars->storable;
    if (can_compute && params->blend_params)
        can_compute = tpars->format->caps & PL_FMT_CAP_READWRITE;

    if (pl_shader_is_compute(sh) && !can_compute) {
        PL_ERR(dp, "Trying to dispatch using a compute shader with a "
               "non-storable or incompatible target texture.");
        goto error;
    } else if (can_compute && limits->compute_queues > limits->fragment_queues) {
        if (sh_try_compute(sh, 16, 16, true, 0))
            PL_TRACE(dp, "Upgrading fragment shader to compute shader.");
    }

    pl_rect2d rc = params->rect;
    if (!pl_rect_w(rc)) {
        rc.x0 = 0;
        rc.x1 = tpars->w;
    }
    if (!pl_rect_h(rc)) {
        rc.y0 = 0;
        rc.y1 = tpars->h;
    }

    int w, h, tw = abs(pl_rect_w(rc)), th = abs(pl_rect_h(rc));
    if (pl_shader_output_size(sh, &w, &h) && (w != tw || h != th))
    {
        PL_ERR(dp, "Trying to dispatch a shader with explicit output size "
               "requirements %dx%d%s using a target rect of size %dx%d.",
               w, h, sh->transpose ? " (transposed)" : "", tw, th);
        goto error;
    }

    int vert_idx = -1;
    const pl_transform2x2 *proj = NULL;
    if (pl_shader_is_compute(sh)) {
        // Translate the compute shader to simulate vertices etc.
        translate_compute_shader(dp, sh, &rc, params);
    } else {
        // Add the vertex information encoding the position
        pl_rect2df vert_rect = {
            .x0 = 2.0 * rc.x0 / tpars->w - 1.0,
            .y0 = 2.0 * rc.y0 / tpars->h - 1.0,
            .x1 = 2.0 * rc.x1 / tpars->w - 1.0,
            .y1 = 2.0 * rc.y1 / tpars->h - 1.0,
        };

        if (sh->transpose) {
            static const pl_transform2x2 transpose_proj = {{{
                { 0, 1 },
                { 1, 0 },
            }}};
            proj = &transpose_proj;
            PL_SWAP(vert_rect.x0, vert_rect.y0);
            PL_SWAP(vert_rect.x1, vert_rect.y1);
        }

        sh_attr_vec2(sh, "position", &vert_rect);
        vert_idx = sh->vas.num - 1;
    }

    // We need to set pl_pass_params.load_target when either blending is
    // enabled or we're drawing to some scissored sub-rect of the texture
    pl_rect2d full = { 0, 0, tpars->w, tpars->h };
    pl_rect2d rc_norm = rc;
    pl_rect2d_normalize(&rc_norm);
    rc_norm.x0 = PL_MAX(rc_norm.x0, 0);
    rc_norm.y0 = PL_MAX(rc_norm.y0, 0);
    rc_norm.x1 = PL_MIN(rc_norm.x1, tpars->w);
    rc_norm.y1 = PL_MIN(rc_norm.y1, tpars->h);
    bool load = params->blend_params || !pl_rect2d_eq(rc_norm, full);
    pl_rect2d rc_full_norm = rc;
    pl_rect2d_normalize(&rc_full_norm);
    if (pl_tex_rgb16f_to_yuv(dp->gpu, params->target) &&
        (load || !pl_rect2d_eq(rc_full_norm, full) || pl_shader_is_compute(sh)))
    {
        PL_ERR(dp, "YUV drawable requires an unblended full-surface raster pass");
        goto error;
    }

    atlas = pl_tex_uv_atlas_anchor(dp->gpu, params->target);
    if (atlas && (atlas->rejected || load || pl_shader_is_compute(sh) ||
        params->blend_params || !pl_rect2d_eq(rc_full_norm, full) ||
        pl_tex_rgb16f_to_yuv(dp->gpu, params->target)))
    {
        atlas->finish(dp->gpu, params->target, false);
        goto error;
    }
    // Atlas runs on the ordinary target and never enters the YUV/ROI machinery.
    struct pl_yuv_compare *compare = atlas ? NULL : pl_tex_yuv_compare(dp->gpu, params->target);
    ident_t compare_words = NULL_IDENT;
    if (compare) {
        if (compare->rejected || load || pl_shader_is_compute(sh) || !pl_rect2d_eq(rc_full_norm, full) ||
            !pl_tex_rgb16f_to_yuv(dp->gpu, params->target) ||
            !compare->baseline || !compare->storage) {
            compare->finish(dp->gpu, params->target, false, false, 0);
            goto error;
        }
        struct pl_buffer_var words = {
            .var = pl_var_uint("mks_capture_words"),
            .layout = { .offset = 0, .stride = sizeof(uint32_t),
                        .size = compare->storage->params.size },
        };
        words.var.dim_a = compare->storage->params.size / sizeof(uint32_t);
        sh_desc(sh, (struct pl_shader_desc) {
            .desc = { .name = "mks_capture", .type = PL_DESC_BUF_STORAGE,
                      .access = PL_DESC_ACCESS_READWRITE },
            .binding.object = compare->storage,
            .memory = PL_MEMORY_COHERENT,
            .buffer_vars = &words, .num_buffer_vars = 1,
        });
        compare_words = sh_ident_unpack(sh->descs.elem[sh->descs.num-1].buffer_vars[0].var.name);
    }

    // Default-off production UV table. Only an un-instrumented ordinary
    // full-surface RGB16F->YUV raster pass may sample through the table; any
    // diagnostic machinery (atlas/compare) or partial draw keeps ordinary
    // generation. The SSBO block and its binding come from this sh_desc
    // descriptor alone; generate_shaders only appends the lookup helper and
    // the coordinate macro. The table only exists on the OpenGL backend;
    // without it every pass is ordinary generation.
    struct pl_uv_atlas_table_params uv_table = {0};
    uint64_t uv_table_generation = 0;
    int uv_table_width = 0, uv_table_height = 0;
    size_t uv_table_words = 0;
    pl_buf uv_table_storage = NULL;
#ifdef PL_HAVE_OPENGL
    bool table_expected =
        !atlas && !compare && !load && !params->blend_params &&
        !pl_shader_is_compute(sh) &&
        pl_tex_rgb16f_to_yuv(dp->gpu, params->target) &&
        params->target->params.w > 0 && params->target->params.h > 0 &&
        pl_opengl_uv_atlas_table_active(dp->gpu);
    uv_table_storage = table_expected
        ? pl_opengl_uv_atlas_table_info(dp->gpu, &uv_table_width, &uv_table_height,
                                        &uv_table_words, &uv_table_generation) : NULL;
    // The table is addressed by gl_FragCoord: a target of any other size
    // would read out of bounds. A mismatch means the geometry changed since
    // seeding, so the table is dead by contract: detach it (the caller
    // observes table_active going false and reseeds).
    if (uv_table_storage &&
        (uv_table_width != params->target->params.w ||
         uv_table_height != params->target->params.h)) {
        pl_opengl_uv_atlas_table_detach(dp->gpu);
        uv_table_storage = NULL;
    }
    if (table_expected && !uv_table_storage) {
        // A healthy table was attached and this is the production fusion
        // pass shape, but the pass cannot sample through the table (no
        // recorded direct-sample tap, geometry drift after the detach
        // above, or the info probe failed). Running the fusion untabled
        // would reintroduce the UV rounding regression the table exists to
        // remove, so fail the dispatch: the caller's fusion failure path
        // re-renders this frame through the ordinary split path and the
        // detached table reseeds on the next eligible frame.
        PL_ERR(dp->gpu, "MKS-FUSION-TABLE event=table-required-unmet ident=%d active=%d",
               sh->sample_uv_ident ? 1 : 0,
               pl_opengl_uv_atlas_table_active(dp->gpu) ? 1 : 0);
        goto error;
    }
#endif
    if (uv_table_storage && uv_table_words >= 3) {
        struct pl_buffer_var tvars = {
            .var = pl_var_uint("mks_cache_words"),
            .layout = { .offset = 0, .stride = sizeof(uint32_t),
                        .size = uv_table_storage->params.size },
        };
        tvars.var.dim_a = uv_table_words;
        sh_desc(sh, (struct pl_shader_desc) {
            .desc = { .name = "mks_cache_block", .type = PL_DESC_BUF_STORAGE,
                      .access = PL_DESC_ACCESS_READONLY },
            .binding.object = uv_table_storage,
            .buffer_vars = &tvars, .num_buffer_vars = 1,
        });
        uv_table.storage = uv_table_storage;
        uv_table.width = uv_table_width;
        uv_table.words = uv_table_words;
        uv_table.generation = uv_table_generation;
        uv_table.words_ident = sh_ident_unpack(sh->descs.elem[sh->descs.num-1].buffer_vars[0].var.name);
    }

    struct pass *pass = finalize_pass(dp, sh, params->target, vert_idx,
                                      params->blend_params, load, NULL, proj, compare, compare_words, atlas,
                                      uv_table.storage ? &uv_table : NULL);

    // Silently return on failed passes
    if (!pass || (!pass->pass && !compare))
        goto error;
    if (compare && !pass->compare_baseline) {
        compare->finish(dp->gpu, params->target, false, false, pass->compare_body_hash);
        goto error;
    }

    struct pl_pass_run_params *rparams = &pass->run_params;

    // Update the descriptor bindings
    for (int i = 0; i < sh->descs.num; i++)
        rparams->desc_bindings[i] = sh->descs.elem[i].binding;

    // Update all of the variables (if needed)
    rparams->num_var_updates = 0;
    for (int i = 0; i < sh->vars.num; i++)
        update_pass_var(dp, pass, &sh->vars.elem[i], &pass->vars[i]);

    // Update the vertex data
    if (rparams->vertex_data) {
        uintptr_t vert_base = (uintptr_t) rparams->vertex_data;
        pl_pass vertex_pass = pass->pass ? pass->pass : pass->compare_baseline;
        size_t stride = vertex_pass->params.vertex_stride;
        for (int i = 0; i < sh->vas.num; i++) {
            const struct pl_shader_va *sva = &sh->vas.elem[i];
            struct pl_vertex_attrib *va = &vertex_pass->params.vertex_attribs[i];

            size_t size = sva->attr.fmt->texel_size;
            uintptr_t va_base = vert_base + va->offset; // use placed offset
            for (int n = 0; n < 4; n++)
                memcpy((void *) (va_base + n * stride), sva->data[n], size);
        }
    }

    // For compute shaders: also update the dispatch dimensions
    if (pl_shader_is_compute(sh)) {
        int width = abs(pl_rect_w(rc)),
            height = abs(pl_rect_h(rc));
        if (sh->transpose)
            PL_SWAP(width, height);
        // Round up to make sure we don't leave off a part of the target
        int block_w = sh->group_size[0],
            block_h = sh->group_size[1],
            num_x   = PL_DIV_UP(width, block_w),
            num_y   = PL_DIV_UP(height, block_h);

        rparams->compute_groups[0] = num_x;
        rparams->compute_groups[1] = num_y;
        rparams->compute_groups[2] = 1;
    } else {
        // Update the scissors for performance
        rparams->scissors = rc_norm;
    }

    // Dispatch the actual shader
    rparams->target = params->target;
    rparams->timer = PL_DEF(params->timer, pass->timer);
    if (atlas) {
        ret = run_atlas_anchor(dp, sh, pass, atlas, params->target);
    } else if (compare) {
        ret = run_compare_passes(dp, sh, pass, compare, params->target);
    } else {
        run_pass(dp, sh, pass);
        ret = true;
#ifdef PL_HAVE_OPENGL
        if (uv_table.storage) {
            // Table-drive receipt: count and (once per generation) log that
            // this draw really sampled through the attached UV table. The
            // caller rejects any fused frame that did not advance the serial.
            pl_opengl_uv_atlas_table_note_pass(dp->gpu);
            static uint64_t last_table_run_generation;
            static pl_dispatch last_table_run_dispatch;
            if (last_table_run_dispatch != dp || last_table_run_generation != uv_table.generation) {
                last_table_run_dispatch = dp;
                last_table_run_generation = uv_table.generation;
                PL_ERR(dp->gpu, "MKS-FUSION-TABLE event=table-pass-ran generation=%"PRIu64,
                       uv_table.generation);
            }
        }
#endif
    }
    // fall through

error:
    if (atlas && !atlas->ended)
        atlas->finish(dp->gpu, params->target, false);
    // Reset the temporary buffers which we use to build the shader
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp); i++)
        pl_str_builder_reset(dp->tmp[i]);

    pl_mutex_unlock(&dp->lock);
    pl_dispatch_abort(dp, params->shader);
    return ret;
}

bool pl_dispatch_compute(pl_dispatch dp, const struct pl_dispatch_compute_params *params)
{
    pl_shader sh = *params->shader;
    bool ret = false;
    pl_mutex_lock(&dp->lock);

    if (sh->failed) {
        PL_ERR(sh, "Trying to dispatch a failed shader.");
        goto error;
    }

    if (!sh->mutable) {
        PL_ERR(dp, "Trying to dispatch non-mutable shader?");
        goto error;
    }

    if (sh->input != PL_SHADER_SIG_NONE) {
        PL_ERR(dp, "Trying to dispatch shader with incompatible signature!");
        goto error;
    }

    if (!pl_shader_is_compute(sh)) {
        PL_ERR(dp, "Trying to dispatch a non-compute shader using "
               "`pl_dispatch_compute`!");
        goto error;
    }

    if (sh->vas.num) {
        if (!params->width || !params->height) {
            PL_ERR(dp, "Trying to dispatch a targetless compute shader that "
                   "uses vertex attributes, this requires specifying the size "
                   "of the effective rendering area!");
            goto error;
        }

        compute_vertex_attribs(dp, sh, params->width, params->height,
                               &(ident_t){0});
    }

    struct pass *pass = finalize_pass(dp, sh, NULL, -1, NULL, false, NULL, NULL, NULL, NULL_IDENT, NULL, NULL);

    // Silently return on failed passes
    if (!pass || !pass->pass)
        goto error;

    struct pl_pass_run_params *rparams = &pass->run_params;

    // Update the descriptor bindings
    for (int i = 0; i < sh->descs.num; i++)
        rparams->desc_bindings[i] = sh->descs.elem[i].binding;

    // Update all of the variables (if needed)
    rparams->num_var_updates = 0;
    for (int i = 0; i < sh->vars.num; i++)
        update_pass_var(dp, pass, &sh->vars.elem[i], &pass->vars[i]);

    // Update the dispatch size
    int groups = 1;
    for (int i = 0; i < 3; i++) {
        groups *= params->dispatch_size[i];
        rparams->compute_groups[i] = params->dispatch_size[i];
    }

    if (!groups) {
        pl_assert(params->width && params->height);
        int block_w = sh->group_size[0],
            block_h = sh->group_size[1],
            num_x   = PL_DIV_UP(params->width, block_w),
            num_y   = PL_DIV_UP(params->height, block_h);

        rparams->compute_groups[0] = num_x;
        rparams->compute_groups[1] = num_y;
        rparams->compute_groups[2] = 1;
    }

    // Dispatch the actual shader
    rparams->timer = PL_DEF(params->timer, pass->timer);
    run_pass(dp, sh, pass);

    ret = true;
    // fall through

error:
    // Reset the temporary buffers which we use to build the shader
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp); i++)
        pl_str_builder_reset(dp->tmp[i]);

    pl_mutex_unlock(&dp->lock);
    pl_dispatch_abort(dp, params->shader);
    return ret;
}

bool pl_dispatch_vertex(pl_dispatch dp, const struct pl_dispatch_vertex_params *params)
{
    pl_shader sh = *params->shader;
    bool ret = false;
    pl_mutex_lock(&dp->lock);

    if (sh->failed) {
        PL_ERR(sh, "Trying to dispatch a failed shader.");
        goto error;
    }

    if (!sh->mutable) {
        PL_ERR(dp, "Trying to dispatch non-mutable shader?");
        goto error;
    }

    if (sh->input != PL_SHADER_SIG_NONE || sh->output != PL_SHADER_SIG_COLOR) {
        PL_ERR(dp, "Trying to dispatch shader with incompatible signature!");
        goto error;
    }

    if (pl_tex_rgb16f_to_yuv(dp->gpu, params->target)) {
        PL_ERR(dp, "YUV drawable does not support custom vertex dispatch");
        goto error;
    }
    const struct pl_tex_params *tpars = &params->target->params;
    if (pl_tex_params_dimension(*tpars) != 2 || !tpars->renderable) {
        PL_ERR(dp, "Trying to dispatch a shader using an invalid target "
               "texture. The target must be a renderable 2D texture.");
        goto error;
    }

    if (pl_shader_is_compute(sh)) {
        PL_ERR(dp, "Trying to dispatch a compute shader using pl_dispatch_vertex.");
        goto error;
    }

    if (sh->vas.num) {
        PL_ERR(dp, "Trying to dispatch a custom vertex shader with already "
               "attached vertex attributes.");
        goto error;
    }

    if (sh->transpose) {
        PL_ERR(dp, "Trying to dispatch a transposed shader using "
               "pl_dispatch_vertex, unlikely to be correct. Erroring as a "
               "safety precaution!");
        goto error;
    }

    int pos_idx = params->vertex_position_idx;
    if (pos_idx < 0 || pos_idx >= params->num_vertex_attribs) {
        PL_ERR(dp, "Vertex position index out of range?");
        goto error;
    }

    // Attach all of the vertex attributes to the shader manually
    sh->vas.num = params->num_vertex_attribs;
    PL_ARRAY_RESIZE(sh, sh->vas, sh->vas.num);
    for (int i = 0; i < params->num_vertex_attribs; i++) {
        ident_t id = sh_fresh(sh, params->vertex_attribs[i].name);
        sh->vas.elem[i].attr = params->vertex_attribs[i];
        sh->vas.elem[i].attr.name = sh_ident_pack(id);
        GLSLP("#define %s "$"\n", params->vertex_attribs[i].name, id);
    }

    // Compute the coordinate projection matrix
    pl_transform2x2 proj = pl_transform2x2_identity;
    switch (params->vertex_coords) {
    case PL_COORDS_ABSOLUTE:
        proj.mat.m[0][0] /= tpars->w;
        proj.mat.m[1][1] /= tpars->h;
        // fall through
    case PL_COORDS_RELATIVE:
        proj.mat.m[0][0] *= 2.0;
        proj.mat.m[1][1] *= 2.0;
        proj.c[0] -= 1.0;
        proj.c[1] -= 1.0;
        // fall through
    case PL_COORDS_NORMALIZED:
        if (params->vertex_flipped) {
            proj.mat.m[1][1] = -proj.mat.m[1][1];
            proj.c[1] += 2.0;
        }
        break;
    }

    struct pass *pass = finalize_pass(dp, sh, params->target, pos_idx,
                                      params->blend_params, true, params, &proj, NULL, NULL_IDENT, NULL, NULL);

    // Silently return on failed passes
    if (!pass || !pass->pass)
        goto error;

    struct pl_pass_run_params *rparams = &pass->run_params;

    // Update the descriptor bindings
    for (int i = 0; i < sh->descs.num; i++)
        rparams->desc_bindings[i] = sh->descs.elem[i].binding;

    // Update all of the variables (if needed)
    rparams->num_var_updates = 0;
    for (int i = 0; i < sh->vars.num; i++)
        update_pass_var(dp, pass, &sh->vars.elem[i], &pass->vars[i]);

    // Update the scissors
    rparams->scissors = params->scissors;
    if (params->vertex_flipped) {
        rparams->scissors.y0 = tpars->h - rparams->scissors.y0;
        rparams->scissors.y1 = tpars->h - rparams->scissors.y1;
    }
    pl_rect2d_normalize(&rparams->scissors);

    // Dispatch the actual shader
    rparams->target = params->target;
    rparams->vertex_count = params->vertex_count;
    rparams->vertex_data = params->vertex_data;
    rparams->vertex_buf = params->vertex_buf;
    rparams->buf_offset = params->buf_offset;
    rparams->index_data = params->index_data;
    rparams->index_fmt = params->index_fmt;
    rparams->index_buf = params->index_buf;
    rparams->index_offset = params->index_offset;
    rparams->timer = PL_DEF(params->timer, pass->timer);
    run_pass(dp, sh, pass);

    ret = true;
    // fall through

error:
    // Reset the temporary buffers which we use to build the shader
    for (int i = 0; i < PL_ARRAY_SIZE(dp->tmp); i++)
        pl_str_builder_reset(dp->tmp[i]);

    pl_mutex_unlock(&dp->lock);
    pl_dispatch_abort(dp, params->shader);
    return ret;
}

void pl_dispatch_abort(pl_dispatch dp, pl_shader *psh)
{
    pl_shader sh = *psh;
    if (!sh)
        return;

    // Free unused memory as early as possible
    sh_deref(sh);

    // Re-add the shader to the internal pool of shaders
    pl_mutex_lock(&dp->lock);
    PL_ARRAY_APPEND(dp, dp->shaders, sh);
    pl_mutex_unlock(&dp->lock);
    *psh = NULL;
}

void pl_dispatch_reset_frame(pl_dispatch dp)
{
    pl_mutex_lock(&dp->lock);

    dp->current_ident = 0;
    dp->current_index++;
    garbage_collect_passes(dp);

    pl_mutex_unlock(&dp->lock);
}

size_t pl_dispatch_save(pl_dispatch dp, uint8_t *out)
{
    return pl_cache_save(pl_gpu_cache(dp->gpu), out, out ? SIZE_MAX : 0);
}

void pl_dispatch_load(pl_dispatch dp, const uint8_t *cache)
{
    pl_cache_load(pl_gpu_cache(dp->gpu), cache, SIZE_MAX);
}
