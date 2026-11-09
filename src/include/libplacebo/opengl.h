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
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBPLACEBO_OPENGL_H_
#define LIBPLACEBO_OPENGL_H_

#include <string.h>

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>

PL_API_BEGIN

// Note on thread safety: The thread safety of `pl_opengl` and any associated
// GPU objects follows the same thread safety rules as the underlying OpenGL
// context. In other words, they must only be called from the thread the OpenGL
// context is current on.

typedef const struct pl_opengl_t {
    pl_gpu gpu;

    // Detected GL version
    int major, minor;

    // List of GL/EGL extensions, provided for convenience
    const char * const *extensions;
    int num_extensions;
} *pl_opengl;

static inline bool pl_opengl_has_ext(pl_opengl gl, const char *ext)
{
    for (int i = 0; i < gl->num_extensions; i++)
        if (!strcmp(ext, gl->extensions[i]))
            return true;
    return false;
}

typedef void (*pl_voidfunc_t)(void);

struct pl_opengl_params {
    // Main gl*GetProcAddr function. This will be used to load all GL/EGL
    // functions. Optional - if unspecified, libplacebo will default to an
    // internal loading logic which should work on most platforms.
    pl_voidfunc_t (*get_proc_addr_ex)(void *proc_ctx, const char *procname);
    void *proc_ctx;

    // Simpler API for backwards compatibility / convenience. (This one
    // directly matches the signature of most gl*GetProcAddr library functions)
    pl_voidfunc_t (*get_proc_addr)(const char *procname);

    // Enable OpenGL debug report callbacks. May have little effect depending
    // on whether or not the GL context was initialized with appropriate
    // debugging enabled.
    bool debug;

    // Allow the use of (suspected) software rasterizers and renderers. These
    // can be useful for debugging purposes, but normally, their use is
    // undesirable when GPU-accelerated processing is expected.
    bool allow_software;

    // Disables the use of compute shaders. Some devices/drivers perform better
    // without them. This may also help prevent image corruption in cases where
    // the driver is misbehaving. Some features may be disabled if this is set.
    bool no_compute;

    // Restrict the maximum allowed GLSL version. (Mainly for testing)
    int max_glsl_version;

    // Optional. Required when importing/exporting dmabufs as textures.
    void *egl_display;
    void *egl_context;

    // Optional callbacks to bind/release the OpenGL context on the current
    // thread. If these are specified, then the resulting `pl_gpu` will have
    // `pl_gpu_limits.thread_safe` enabled, and may therefore be used from any
    // thread without first needing to bind the OpenGL context.
    //
    // If the user is re-using the same OpenGL context in non-libplacebo code,
    // then these callbacks should include whatever synchronization is
    // necessary to prevent simultaneous use between libplacebo and the user.
    bool (*make_current)(void *priv);
    void (*release_current)(void *priv);
    void *priv;

    // Explicit default-off numeric comparison diagnostics. On GLES 3.1+, query
    // real core SSBO limits when legacy extension aliases did not expose them.
    // Requires the configured EGL owner current, all capture entry points and
    // an error-free query. The measured limit is private to fixed capture
    // buffers; renderer-visible limits and shader planning remain unchanged.
    bool diagnostic_yuv_compare_caps;
    // M2' frame-timing owner: disable every internal GL timer path so the
    // frame-timing owner is the only elapsed-query user in this context.
    // Requires PL_HAVE_MKS_DISABLE_TIMERS on the mpv side; without the macro
    // the field is ignored and the owner must refuse to arm.
    bool mks_disable_timers;
};
#define PL_HAVE_MKS_YUV_COMPARE_CAPS 1
#define PL_HAVE_MKS_DISABLE_TIMERS 1

// Default/recommended parameters
#define pl_opengl_params(...) (&(struct pl_opengl_params) { __VA_ARGS__ })
PL_API extern const struct pl_opengl_params pl_opengl_default_params;

// Creates a new OpenGL renderer based on the given parameters. This will
// internally use whatever platform-defined mechanism (WGL, X11, EGL) is
// appropriate for loading the OpenGL function calls, so the user doesn't need
// to pass in a `getProcAddress` callback. If `params` is left as NULL, it
// defaults to `&pl_opengl_default_params`. The context must be active when
// calling this function, and must remain active whenever calling any
// libplacebo function on the resulting `pl_opengl` or `pl_gpu`.
//
// Note that creating multiple `pl_opengl` instances from the same OpenGL
// context is undefined behavior.
PL_API pl_opengl pl_opengl_create(pl_log log, const struct pl_opengl_params *params);

// All resources allocated from the `pl_gpu` contained by this `pl_opengl` must
// be explicitly destroyed by the user before calling `pl_opengl_destroy`.
PL_API void pl_opengl_destroy(pl_opengl *gl);

// For a `pl_gpu` backed by `pl_opengl`, this function can be used to retrieve
// the underlying `pl_opengl`. Returns NULL for any other type of `gpu`.
PL_API pl_opengl pl_opengl_get(pl_gpu gpu);

struct pl_opengl_framebuffer {
    // ID of the framebuffer, or 0 to use the context's default framebuffer.
    int id;

    // If true, then the framebuffer is assumed to be "flipped" relative to
    // normal GL semantics, i.e. set this to `true` if the first pixel is the
    // top left corner.
    bool flipped;
};

struct pl_opengl_swapchain_params {
    // Set this to the platform-specific function to swap buffers, e.g.
    // glXSwapBuffers, eglSwapBuffers etc. This will be called internally by
    // `pl_swapchain_swap_buffers`. Required, unless you never call that
    // function.
    void (*swap_buffers)(void *priv);

    // Initial framebuffer description. This can be changed later on using
    // `pl_opengl_swapchain_update_fb`.
    struct pl_opengl_framebuffer framebuffer;

    // Attempt forcing a specific latency. If this is nonzero, then
    // `pl_swapchain_swap_buffers` will wait until fewer than N frames are "in
    // flight" before returning. Setting this to a high number generally
    // accomplished nothing, because the OpenGL driver typically limits the
    // number of buffers on its own. But setting it to a low number like 2 or
    // even 1 can reduce latency (at the cost of throughput).
    int max_swapchain_depth;

    // Arbitrary user pointer that gets passed to `swap_buffers` etc.
    void *priv;

    // Explicit diagnostic-only opt-in for abort_frame_no_gl. Default false.
    // Only the live thread that successfully starts a frame may abort it.
    bool diagnostic_abort_frame_no_gl;
};

#define pl_opengl_swapchain_params(...) (&(struct pl_opengl_swapchain_params) { __VA_ARGS__ })

// Creates an instance of `pl_swapchain` tied to the active context.
// Note: Due to OpenGL semantics, users *must* call `pl_swapchain_resize`
// before attempting to use this swapchain, otherwise calls to
// `pl_swapchain_start_frame` will fail.
PL_API pl_swapchain pl_opengl_create_swapchain(pl_opengl gl,
                                               const struct pl_opengl_swapchain_params *params);

// Update the framebuffer description. After calling this function, users
// *must* call `pl_swapchain_resize` before attempting to use the swapchain
// again, otherwise calls to `pl_swapchain_start_frame` will fail.
PL_API void pl_opengl_swapchain_update_fb(pl_swapchain sw,
                                          const struct pl_opengl_framebuffer *fb);

#define PL_HAVE_MKS_YUV_ABORT_FRAME 1
// Diagnostic emergency exit after EGL owner restoration failed. Requires the
// explicit swapchain opt-in above, a currently open frame, and the same live
// thread that successfully called start_frame. Returns false without touching
// frame state/locks for a foreign backend, thread, or closed frame. The caller
// must keep the swapchain alive and must not concurrently destroy it.
// This only clears frame state and releases its CPU mutex: it makes no GL call,
// context release, flush, submit, swap, or deletion, and does not undo previously
// issued GPU work. The aborted drawable must never subsequently be swapped.
PL_API bool pl_opengl_swapchain_abort_frame_no_gl(pl_swapchain sw);

#define PL_HAVE_MKS_YUV_OUTPUT 1

// Experimental render-only encoding of an externally owned, legal YUV drawable.
// The shader computes logical RGB16F, then emulates its store and encodes FULL
// BT.2020 NCL YUV. This is not an ordinary RGBA framebuffer or YUV sampler.
enum pl_opengl_output_encoding {
    PL_OPENGL_OUTPUT_NORMAL = 0,
    PL_OPENGL_OUTPUT_RGB16F_TO_BT2020_FULL_YUV,
};

struct pl_opengl_wrap_params {
    // The GLuint texture object itself. Optional. If no texture is provided,
    // then only the opaque framebuffer `fbo` will be wrapped, leaving the
    // resulting `pl_tex` object with some operations (such as sampling) being
    // unsupported.
    unsigned int texture;

    // The GLuint associated framebuffer. Optional. If this is not specified,
    // then libplacebo will attempt creating a framebuffer from the provided
    // texture object (if possible).
    //
    // Note: As a special case, if neither a texture nor an FBO are provided,
    // this is equivalent to wrapping the OpenGL default framebuffer (id 0).
    unsigned int framebuffer;

    // The image's dimensions (unused dimensions must be 0)
    int width;
    int height;
    int depth;

    // Texture-specific fields:
    //
    // Note: These are only relevant if `texture` is provided.

    // The GLenum for the texture target to use, e.g. GL_TEXTURE_2D. Optional.
    // If this is left as 0, the target is inferred from the number of
    // dimensions. Users may want to set this to something specific like
    // GL_TEXTURE_EXTERNAL_OES depending on the nature of the texture.
    unsigned int target;

    // The texture's GLint sized internal format (e.g. GL_RGBA16F). Required.
    int iformat;

    // Overrides the sampler type used to sample from this texture. The
    // default (PL_SAMPLER_NORMAL) infers the sampler type from the texture
    // `target`, and is usually what you want.
    //
    // Setting this to PL_SAMPLER_EXTERNAL_YUV selects raw Y/Cb/Cr sampling
    // of a GL_TEXTURE_EXTERNAL_OES via GL_EXT_YUV_target (requires GLES 3.0
    // and the GL_EXT_YUV_target extension; wrapping fails otherwise). In this
    // mode, the sampler returns the raw, driver-upsampled YUV values of the
    // underlying external image. This bypasses libplacebo's own chroma
    // resampling and chroma location handling, and it is the caller's
    // responsibility to describe the resulting values with an appropriate
    // `pl_color_repr` (e.g. levels and bit depth).
    enum pl_sampler_type sampler_type;

    // Default NORMAL leaves all existing wrapping behavior unchanged. The YUV
    // mode requires a framebuffer-only GLES 3.1+ EXT_YUV_target drawable, with
    // iformat=GL_RGBA16F declaring the LOGICAL intermediate format used by the
    // renderer. The caller guarantees that framebuffer is physically YUV and
    // covers the whole drawable with opaque content. Only full-surface raster
    // pl_dispatch_finish without blending is supported; clear, sample, storage,
    // transfer, blit and custom vertex dispatch are unavailable. No ownership is
    // transferred. The caller must preserve an ordinary RGB fallback target.
    enum pl_opengl_output_encoding output_encoding;
};

#define pl_opengl_wrap_params(...) (&(struct pl_opengl_wrap_params) { __VA_ARGS__ })

// Wraps an external OpenGL object into a `pl_tex` abstraction. Due to the
// internally synchronized nature of OpenGL, no explicit synchronization
// is needed between libplacebo `pl_tex_` operations, and host accesses to
// the texture. Wrapping the same OpenGL texture multiple times is permitted.
// Note that this function transfers no ownership.
//
// This wrapper can be destroyed by simply calling `pl_tex_destroy` on it,
// which will *not* destroy the user-provided OpenGL texture or framebuffer.
//
// This function may fail, in which case it returns NULL.
PL_API pl_tex pl_opengl_wrap(pl_gpu gpu, const struct pl_opengl_wrap_params *params);

// Default-off, fixed-frame full-screen numerical anchor. This is not a cache
// or production optimization. The caller must cancel before destroying the GPU.
#define PL_HAVE_MKS_UV_ATLAS_ANCHOR 1
#define PL_OPENGL_UV_ATLAS_ANCHOR_MAX_PIXELS 4194304u
typedef struct pl_opengl_uv_atlas_anchor_t *pl_opengl_uv_atlas_anchor;
struct pl_opengl_uv_atlas_anchor_params {
    uint64_t capture_id, generation;
    // Production seeding: run the A0/record/A0 triple with zero CPU readback
    // and no replay-equality accounting. The fence alone gates completion;
    // numerical validation remains the diagnostic mode's job. Once the seed
    // transaction finishes cleanly it is sealed: errors raised later by
    // rendering outside the transaction do not invalidate it (the diagnostic
    // mode's global error-epoch equality is not retried after finish).
    bool production_seed;
};
struct pl_opengl_uv_atlas_anchor_result {
    uint64_t capture_id, generation;
    int width, height;
    size_t pixels, rgba_words, atlas_words; // rgba_words is per stage, not total
    bool draw_ok[3], readback_ok[3], half_exact[3];
    uint64_t program_tokens[3]; // original A0, atlas A, same original A0
    uint64_t draw_epoch_before[3], draw_epoch_after[3];
    uint64_t readback_epoch_before[3], readback_epoch_after[3], finish_epoch;
    unsigned int framebuffer[3], read_format, read_type;
    bool source_abi_ok, fence_signaled, atlas_complete, anchor_equal;
    size_t mismatches_before_atlas, mismatches_before_after;
    size_t bad_half_components[3], bad_atlas_pixels;
};
enum pl_opengl_uv_atlas_anchor_status {
    PL_OPENGL_UV_ATLAS_ANCHOR_FAILED = -1,
    PL_OPENGL_UV_ATLAS_ANCHOR_PENDING = 0,
    PL_OPENGL_UV_ATLAS_ANCHOR_COMPLETE = 1,
};
// Target must be the ordinary RGB16F FBO, kept alive through the one render.
// Its destruction detaches/invalidates the session; the independent handle
// remains safe to poll/cancel without dereferencing the destroyed texture.
PL_API pl_opengl_uv_atlas_anchor pl_opengl_uv_atlas_anchor_arm(pl_gpu gpu,
    pl_tex target, const struct pl_opengl_uv_atlas_anchor_params *params);
PL_API enum pl_opengl_uv_atlas_anchor_status pl_opengl_uv_atlas_anchor_poll(
    pl_opengl_uv_atlas_anchor anchor, struct pl_opengl_uv_atlas_anchor_result *result);
// Only a fully successful original A0 final replay may supply this receipt.
PL_API bool pl_opengl_uv_atlas_anchor_baseline_ok(pl_opengl_uv_atlas_anchor anchor);
// Borrowed, native uint32 bit-pattern arrays remain valid until cancellation.
// RGB is legal RGBA/FLOAT readback of decoded RGB16F numerical values, not
// physical half storage bytes. Atlas is 3 words/pixel Ubits,Vbits,count. Count
// high bit flags an invalid sampling site/UV; only exactly 1 is complete.
PL_API bool pl_opengl_uv_atlas_anchor_data(pl_opengl_uv_atlas_anchor anchor,
    const uint32_t **a0_before, const uint32_t **atlas_a,
    const uint32_t **a0_after, const uint32_t **atlas);
PL_API bool pl_opengl_uv_atlas_anchor_sources(pl_opengl_uv_atlas_anchor anchor,
    const char **a0_fragment, const char **atlas_fragment,
    const char **vertex, const char **original_body);
// ABI is the same strict host quad layout used by quad_payloads below.
struct pl_opengl_yuv_compare_quad_layout;
PL_API bool pl_opengl_uv_atlas_anchor_vertices(pl_opengl_uv_atlas_anchor anchor,
    const uint8_t **payload, struct pl_opengl_yuv_compare_quad_layout *layout);
// False retains the handle/resources for owner-context retry, with no foreign
// GL deletion. No implicit target access after detach; cancel before GPU uninit.
PL_API bool pl_opengl_uv_atlas_anchor_cancel(pl_opengl_uv_atlas_anchor *anchor);

// Serialize these handles on the owning renderer thread; destroy before GPU.
// Default-off two-PTS consumer diagnostic. Never automatically rebuilds or
// presents B. B0 is a source/ABI anchor only, not an uninstrumented YUV readback.
#define PL_HAVE_MKS_UV_ATLAS_READ 1
#define PL_OPENGL_UV_ATLAS_MATERIAL_VARS 16
#define PL_OPENGL_UV_ATLAS_MATERIAL_BYTES 4096
typedef struct pl_opengl_uv_atlas_cache_t *pl_opengl_uv_atlas_cache;
typedef struct pl_opengl_uv_atlas_read_t *pl_opengl_uv_atlas_read;
// Complete restricted material. No UBO, push constants or specialization
// constants are supported. Object names and pixel contents are not in the key.
struct pl_opengl_uv_atlas_material {
    int input_width, input_height, sampler_type, sample_mode, address_mode;
    int format_type, num_components, component_depth[4], host_bits[4], sample_order[4];
    char format_name[32];
    uint32_t num_variables, uniform_bytes;
    char uniform_names[PL_OPENGL_UV_ATLAS_MATERIAL_VARS][32];
    uint32_t uniform_offsets[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint32_t uniform_sizes[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint32_t uniform_types[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint32_t uniform_dim_v[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint32_t uniform_dim_m[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint32_t uniform_dim_a[PL_OPENGL_UV_ATLAS_MATERIAL_VARS];
    uint8_t uniforms[PL_OPENGL_UV_ATLAS_MATERIAL_BYTES];
};
struct pl_opengl_uv_atlas_cache_info {
    uint64_t cache_id, seed_capture_id, generation;
    int width, height;
    size_t pixels, atlas_words;
    unsigned int storage_buffer;
    uint64_t storage_offset, storage_size;
};
struct pl_opengl_uv_atlas_read_params {
    uint64_t capture_id, generation;
    unsigned int ordinal; // exactly 1 then 2, same cache, no replacement
};
enum pl_opengl_uv_atlas_read_reject {
    PL_OPENGL_UV_ATLAS_READ_REJECT_NONE=0,
    PL_OPENGL_UV_ATLAS_READ_REJECT_SIGNATURE=1,
    PL_OPENGL_UV_ATLAS_READ_REJECT_UNSUPPORTED=2,
    PL_OPENGL_UV_ATLAS_READ_REJECT_INPUT_IDENTITY=3,
    PL_OPENGL_UV_ATLAS_READ_REJECT_ALLOCATION_OR_COMPILE=4,
    PL_OPENGL_UV_ATLAS_READ_REJECT_GL_OR_OWNER=5,
};
struct pl_opengl_uv_atlas_read_result {
    uint64_t capture_id, cache_id, seed_capture_id, generation;
    unsigned int ordinal;
    int width, height;
    size_t pixels, rgba_words, probe_words; // each probe stage is 4 words/pixel
    bool signature_match, draw_ok[3], readback_ok[3], baseline_ok, fence_signaled;
    uint64_t program_tokens[3]; // A0, same probe program for UV and half
    uint64_t b0_program_token; // created, never drawn: source anchor only
    uint64_t seed_input_texture_token, input_texture_token;
    uint64_t draw_epoch_before[3], draw_epoch_after[3];
    uint64_t readback_epoch_before[3], readback_epoch_after[3], finish_epoch;
    unsigned int probe_modes[2]; // actual uniform submissions: 0 UV, 1 half
    size_t bad_uv_pixels, bad_half_pixels;
    bool uv_equal, half_equal;
    unsigned int reject_reason; // 0 none; 1 signature; 2 unsupported; 3 input identity; 4 allocation/compile; 5 GL/owner
};
// Success consumes *anchor; failure leaves it owned by the caller.
PL_API pl_opengl_uv_atlas_cache pl_opengl_uv_atlas_promote(pl_opengl_uv_atlas_anchor *anchor);
PL_API bool pl_opengl_uv_atlas_cache_info(pl_opengl_uv_atlas_cache cache,
    struct pl_opengl_uv_atlas_cache_info *info, const uint32_t **atlas);
PL_API bool pl_opengl_uv_atlas_cache_destroy(pl_opengl_uv_atlas_cache *cache);
// RGB and YUV wrappers must stay alive through the one render. Arm never
// replaces a previously armed read; ordinal 2 requires completed ordinal 1.
PL_API pl_opengl_uv_atlas_read pl_opengl_uv_atlas_read_arm(pl_opengl_uv_atlas_cache cache,
    pl_tex rgb_target, pl_tex yuv_target, const struct pl_opengl_uv_atlas_read_params *params);
PL_API enum pl_opengl_uv_atlas_anchor_status pl_opengl_uv_atlas_read_poll(
    pl_opengl_uv_atlas_read read, struct pl_opengl_uv_atlas_read_result *result);
PL_API bool pl_opengl_uv_atlas_read_baseline_ok(pl_opengl_uv_atlas_read read);
PL_API bool pl_opengl_uv_atlas_read_data(pl_opengl_uv_atlas_read read,
    const uint32_t **a0, const uint32_t **uv, const uint32_t **half);
PL_API bool pl_opengl_uv_atlas_read_sources(pl_opengl_uv_atlas_read read,
    const char **a0, const char **b0, const char **probe, const char **vertex, const char **body);
PL_API bool pl_opengl_uv_atlas_read_signature(pl_opengl_uv_atlas_read read,
    const struct pl_opengl_uv_atlas_material **seed, const struct pl_opengl_uv_atlas_material **current,
    const uint8_t **vertices, struct pl_opengl_yuv_compare_quad_layout *layout);
PL_API bool pl_opengl_uv_atlas_read_cancel(pl_opengl_uv_atlas_read *read);

// Default-off production UV atlas table. Attaching a promoted, non-invalidated
// cache makes every subsequently *generated* RGB16F->YUV final pass whose
// target size exactly matches the table sample through the archived UV table
// (readonly SSBO) instead of recomputing coordinates, eliminating the
// cross-pass rounding divergence this chain was validated against (schema10).
// The cache must stay alive while attached; the caller detaches before
// destroying it, and the table never outlives the GPU that seeded it. Pass
// generation is per-attach, so cached ordinary passes are never reused for
// table passes or vice versa. Identity scope: a target-size mismatch detaches
// the table implicitly and FAILS that pass generation — the caller's fusion
// failure path must re-render the frame through the ordinary split path; an
// attached table with no recorded direct-sample tap likewise fails rather
// than fusing untabled. Same-size source-material or sampler-configuration
// drift is NOT re-checked here and must be gated by the caller. No production
// shader path is altered while nothing is attached.
PL_API bool pl_opengl_uv_atlas_table_attach(pl_gpu gpu,
    pl_opengl_uv_atlas_cache cache);
PL_API void pl_opengl_uv_atlas_table_detach(pl_gpu gpu);
// True when a live table is attached and the last identity check passed.
PL_API bool pl_opengl_uv_atlas_table_active(pl_gpu gpu);
// Monotonic serial of passes that actually sampled through the table. The
// caller verifies it advanced after every fused frame it presents; a fused
// frame that did not advance the serial sampled untabled and must be
// rejected (re-rendered through the ordinary path).
PL_API uint64_t pl_opengl_uv_atlas_table_pass_serial(pl_gpu gpu);
// Internal receipt hook used by dispatch after a table-driven draw ran.
PL_API void pl_opengl_uv_atlas_table_note_pass(pl_gpu gpu);
// Borrowed view of the attached table for pass generation: returns the
// readonly storage buffer and its geometry, or NULL when no live table is
// attached. `generation` (optional out) changes on every attach/detach so
// cached generated passes never mix table epochs.
PL_API pl_buf pl_opengl_uv_atlas_table_info(pl_gpu gpu, int *width, int *height,
                                             size_t *words, uint64_t *generation);
// The seed-time final-tap material identity of the attached table, or NULL.
// Dispatch compares the current pass's tap texture against it and refuses to
// fuse on any drift (dimensions, sampler mode, format class), so a renderer
// path change — e.g. a runtime peak-detect option flipping the final tap
// onto a single-frame cache FBO — can never consume a table recorded for
// another sampling geometry.
PL_API const struct pl_opengl_uv_atlas_material *pl_opengl_uv_atlas_table_seed_material(pl_gpu gpu);
// The seeding pass's four corner UV values (float[8], bit-exact archive of
// the actual mapping numerics), or NULL. Dispatch byte-compares the current
// pass's corner UVs against these; any drift — including same-texture rect
// changes — refuses fusion, closing the mapping identity end to end.
PL_API const float *pl_opengl_uv_atlas_table_seed_uv(pl_gpu gpu);

// Experimental success acknowledgement for a newly wrapped YUV output. True
// only after exactly one backend draw completed without any sticky GPU failure
// or currently pending GL errors. Call only after successful rendering, before
// committing the drawable. Do not reuse a wrapper across frames. False requires
// rendering this frame to the ordinary RGB target and performing its old YUV
// pass. This checks command submission/GL errors, not optical equivalence or GPU
// completion (no finish/wait is introduced).
PL_API bool pl_opengl_yuv_output_ok(pl_gpu gpu, pl_tex target);

// Experimental per-render failure receipt. Obtain immediately before and after
// rendering (and again after flushing). Both calls must succeed and return the
// same epoch; renderer success is also required. Existing sticky GPU failure is
// deliberately NOT reset: an older failed optional YUV draw need not prevent a
// later, independently successful ordinary RGB fallback from being presented.
// The query first checks pending GL errors, so previously consumed and newly
// pending errors are both represented. A failed MAKE_CURRENT returns false.
PL_API bool pl_opengl_error_epoch(pl_gpu gpu, uint64_t *out_epoch);

// Default-off diagnostic: one final dispatch draws candidate YUV, then the
// exact same generated shader/input to the borrowed baseline RGB16F target.
// The caller must present only baseline, keep both targets alive until cancel,
// and never use this transaction for production fusion receipts/performance.
// All handles must be live objects belonging to this same OpenGL GPU. The
// baseline must be a distinct physical RGB16F FBO, not another drawable alias;
// the library checks distinct nonzero FBO ids but cannot prove storage aliasing.
#define PL_HAVE_MKS_YUV_COMPARE 1
struct pl_opengl_yuv_compare_params {
    uint64_t capture_id, generation;
    pl_tex baseline;
    int x, y, width, height; // ROI in unchanged GL framebuffer coordinates
    bool repeat; // explicit B1/A1/B2/A2 same-program diagnostic, default false
    bool sample_tap; // independent off, requires repeat and one proven direct site
    bool invariant_b; // independent off, requires repeat+tap, modifies only B vertex
    bool uv_oracle; // independent off, requires repeat+tap; excludes invariant_b
    bool uv_oracle_use_a; // runtime B uniform: false=own UV, true=corresponding A UV
    bool uv_oracle_probe; // independent off, B-only dependency sidecar; requires uv_oracle
    bool uv_oracle_single_pair; // independent off; requires oracle+probe+repeat+tap
    bool quad_order; // explicit B-only quad payload framework; requires single and use_a=false
    bool quad_order_reverse; // false identity, true complete records [3,2,1,0]
};
#define PL_HAVE_MKS_YUV_COMPARE_QUAD_ORDER 1
#define PL_HAVE_MKS_YUV_COMPARE_ORACLE_SINGLE_PAIR 1
#define PL_HAVE_MKS_YUV_COMPARE_ORACLE_PROBE 1
#define PL_HAVE_MKS_YUV_COMPARE_UV_ORACLE 1
#define PL_HAVE_MKS_YUV_COMPARE_INVARIANT_B 1
#define PL_HAVE_MKS_YUV_COMPARE_SAMPLE_TAP 1
#define PL_HAVE_MKS_YUV_COMPARE_REPEAT 1
struct pl_opengl_yuv_compare_result {
    uint64_t capture_id, generation, shader_body_hash;
    bool candidate_ok, baseline_ok;
    size_t words; // total words across one or two consecutive pair chunks
    unsigned int repetitions; // 1: B1,A1; 2: B1,A1,B2,A2; oracle: A1,B1[,A2,B2]
    bool draw_ok[4];
    uint64_t program_tokens[4]; // process-local immutable pl_pass identities
    uint64_t draw_epoch_before[4], draw_epoch_after[4];
    bool sample_tap;
    unsigned int words_per_pixel; // 14 control, 26 sample-tap
    unsigned int sample_sites; // 0 control, exactly 1 proven static tap site
    uint64_t sample_tap_body_hash;
    bool invariant_b;
    char invariant_uv_identifier[8]; // actual generated UV identifier, empty when off
    bool uv_oracle, uv_oracle_use_a;
    bool uv_oracle_probe;
    bool uv_oracle_single_pair;
    size_t uv_oracle_probe_words; // one or two complete ROI chunks, 10 uints/pixel
    bool quad_order, quad_order_reverse;
    unsigned int uv_oracle_barriers; // exactly two successful A-to-B SSBO barriers
    bool uv_oracle_barrier_ok[2];
    uint64_t uv_oracle_barrier_epoch_before[2], uv_oracle_barrier_epoch_after[2];
};
// Exact submitted host-record layout, bounded diagnostic-only ABI. Only
// float32 attributes are supported; unused depth entries retain format values.
#define PL_OPENGL_YUV_COMPARE_QUAD_MAX_ATTRIBS 16
struct pl_opengl_yuv_compare_quad_attrib {
    char name[32], format_name[32], glsl_type[16];
    enum pl_fmt_type fmt_type;
    int location, num_components, component_depth[4], host_bits[4], sample_order[4];
    bool opaque;
    int num_planes;
    size_t offset, texel_size;
};
struct pl_opengl_yuv_compare_quad_layout {
    enum pl_prim_type primitive;
    bool indexed;
    int vertex_count, num_attributes, position_attrib;
    size_t stride, byte_size;
    pl_rect2d viewport, scissors; // effective pl_pass_run-normalized full rectangles
    struct pl_opengl_yuv_compare_quad_attrib attributes[PL_OPENGL_YUV_COMPARE_QUAD_MAX_ATTRIBS];
};
enum pl_opengl_yuv_compare_status {
    PL_OPENGL_YUV_COMPARE_FAILED = -1,
    PL_OPENGL_YUV_COMPARE_PENDING = 0,
    PL_OPENGL_YUV_COMPARE_COMPLETE = 1,
};
// Each ROI pixel occupies 14 uint32 words: B count/preRGB3/halfRGB3/YUV3,
// A count/preRGB3. Float fields are bit patterns; accept only both counts == 1.
// Repeat returns two consecutive full ROI chunks, not 28 interleaved words per
// pixel. The same B/A programs are reused and one final fence covers all draws.
// Only the last A may supply the RGB receipt. A1's stored RGB/CSC is not captured.
// Sample-tap adds B UV2/sampleRGB3/site-count at 14..19, A at 20..25.
// Each private invocation site-count must be 1, as must both fragment counts.
// UV-oracle uses A1/barrier/B1/A2/barrier/B2 with the same 26-word chunks.
// Both control/oracle share B program/source; only its runtime int uniform varies.
// A is unchanged sample-tap baseline. B reads A UV only within ROI, with A pixel
// and site counts exactly one; missing/invalid A data makes B site-count zero.
// Only A2 may be presented, after original CSC fully covers B2; all draws and
// both barriers must succeed. This is not a production or performance path.
// No implicit enable property exists. Arm requires diagnostic_yuv_compare_caps
// successfully initialized at context creation and a fresh special wrapper,
// the configured EGL display/context, and non-null draw/read surfaces. Every
// capture operation requires this exact owner current (MAKE_CURRENT alone may
// only lock). A foreign owner invalidates the capture without issuing GL calls.
// Foreign cancel retains resources for retry; foreign target destruction skips
// GL deletion and leaks its diagnostic wrapper rather than touching foreign GL.
PL_API bool pl_opengl_yuv_compare_arm(pl_gpu gpu, pl_tex target,
    const struct pl_opengl_yuv_compare_params *params);
PL_API enum pl_opengl_yuv_compare_status pl_opengl_yuv_compare_poll(
    pl_gpu gpu, pl_tex target, uint32_t *words, size_t capacity,
    struct pl_opengl_yuv_compare_result *result);
// B-only sidecar over the same completion fence and immutable capture owner.
// Ten uint words/pixel, one or two complete ROI chunks: atomic record-fragment count,
// helper-call count, actual sample-site execution count, observed A-fragment and
// A-site counts, observed A-UV raw bits x/y, reason mask, actual dependency bool,
// selected source (0 own/control, 1 A/oracle, 2 fallback). Reason: 1 outside ROI
// (A fields not read, zero initialized), 2 A-fragment !=1, 4 A-site !=1, 8 UV not
// finite. The first three counts must be one. Single-pair uses only A1 then
// one barrier then B1, one main/probe chunk, and A1 original CSC presentation.
// All two draws and the one barrier must succeed; no repeat-program claim. Original 26-word B-site count
// continues to mean valid dependency, not actual site execution. Invalid data
// is diagnostic evidence only and never numerical/causal acceptance.
PL_API enum pl_opengl_yuv_compare_status pl_opengl_yuv_compare_probe_poll(
    pl_gpu gpu, pl_tex target, uint32_t *words, size_t capacity);
// Borrowed actual A/B host payload copies used by this capture, valid until
// cancel/target destruction. The B data is an independent full-record identity
// or [3,2,1,0] permutation; A original data, shader, viewport and ABI are unchanged.
// Only a proven owned four-vertex unindexed host triangle strip is supported.
// Getter requires the same capture EGL owner and successful final draw receipt.
PL_API bool pl_opengl_yuv_compare_quad_payloads(pl_gpu gpu, pl_tex target,
    const uint8_t **baseline, const uint8_t **candidate,
    struct pl_opengl_yuv_compare_quad_layout *layout);
// Receipt for the last A RGB draw. Repeat requires all four draws to succeed;
// oracle single-pair requires both A1/B1 and its one barrier to succeed.
// the original single-pair mode remains independent of B success. It drains
// pending errors and rejects every error_epoch change since A's finish/flush,
// including errors consumed by other APIs. Call after all caller flushes.
// The caller still must confirm its original CSC and single swap separately.
PL_API bool pl_opengl_yuv_compare_baseline_ok(pl_gpu gpu, pl_tex target);
// Borrowed generated source pointers remain valid until cancel/target destroy.
PL_API bool pl_opengl_yuv_compare_sources(pl_gpu gpu, pl_tex target,
    const char **candidate, const char **baseline, const char **vertex);
// Sample-tap body snippets (no version/main): borrowed until cancel/destroy.
// Original is finalized before instrumentation; tap is contained in both actual
// shader sources, except UV-oracle where it remains A0-only (B has its own body). Instrumentation may change compilation, so this is a separate
// diagnostic, not proof of an uninstrumented shader's numerical behavior.
PL_API bool pl_opengl_yuv_compare_bodies(pl_gpu gpu, pl_tex target,
    const char **original_body, const char **tap_body);
// Independent actual B/A0 vertex sources for the B-only invariant experiment.
// The ordinary sources() vertex remains A0. Borrowed until cancel/destroy.
PL_API bool pl_opengl_yuv_compare_vertices(pl_gpu gpu, pl_tex target,
    const char **candidate_vertex, const char **baseline_vertex);
PL_API void pl_opengl_yuv_compare_cancel(pl_gpu gpu, pl_tex target);

// Analogous to `pl_opengl_wrap`, this function takes any `pl_tex` (including
// ones created by `pl_tex_create`) and unwraps it to expose the underlying
// OpenGL texture to the user. Note that this function transfers no ownership,
// i.e. the texture object and framebuffer shall not be destroyed by the user.
//
// Returns the OpenGL texture. `out_target` and `out_iformat` will be updated
// to hold the target type and internal format, respectively. (Optional)
//
// For renderable/blittable textures, `out_fbo` will be updated to the ID of
// the framebuffer attached to this texture, or 0 if there is none. (Optional)
PL_API unsigned int pl_opengl_unwrap(pl_gpu gpu, pl_tex tex, unsigned int *out_target,
                                     int *out_iformat, unsigned int *out_fbo);

PL_API_END

#endif // LIBPLACEBO_OPENGL_H_
