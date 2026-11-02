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

/* One-frame numerical anchor helpers; no ordinary shader uses this transform. */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#ifndef MKS_ATLAS_NO_TRANSFORM
#include "sample_tap.h"
#endif

static inline bool mks_atlas_clean_a0(const char *source)
{
    if (!source || strstr(source, "mks_") || strstr(source, "MKS_")) return false;
    for (const char *p=source; (p=strstr(p,"buffer")); p+=6) {
        bool left=p==source || !(isalnum((unsigned char)p[-1]) || p[-1]=='_');
        bool right=!(isalnum((unsigned char)p[6]) || p[6]=='_');
        if (left && right) return false; // conservative: never accept a storage block
    }
    return true;
}

static inline bool mks_atlas_half_exact(uint32_t raw)
{
    uint32_t mag = raw & 0x7fffffffu;
    if (!mag) return true; // preserve either signed zero
    unsigned int e = mag >> 23;
    if (e < 103 || mag > 0x477fe000u) return false;
    uint32_t mask = e < 113 ? (1u << (126 - e)) - 1u : 0x1fffu;
    return !(mag & mask);
}

#ifndef MKS_ATLAS_NO_TRANSFORM
// Insert-only transform of the actual unmodified A0 source. Removing exactly
// the three marker spans including END's LF reconstructs A0 byte for byte.
static inline char *mks_atlas_fragment(const char *source, const char *body,
                                int width, int height)
{
    struct mks_sample_tap_site site;
    if (!mks_atlas_clean_a0(source) || !body || !*body || width < 1 || height < 1 ||
        (size_t) width > 4194304u / (size_t) height || strstr(source, "mks_atlas_") ||
        strstr(source, "MKS_ATLAS_") || !mks_sample_tap_find(body, &site))
        return NULL;
    int version=0, consumed=0;
    if (sscanf(source, "#version %d es%n", &version, &consumed) != 1 ||
        !consumed || source[consumed] != '\n' || version < 310)
        return NULL;
    const char *start = strstr(source, body);
    size_t len = strlen(source);
    if (!start || strstr(start + strlen(body), body) || !len || source[len-1] != '}' ||
        !strstr(source, "void main() {\n") || start + site.start >= source + len - 1)
        return NULL;
    const char *globals =
        "// MKS_ATLAS_GLOBALS_BEGIN\n"
        "layout(std430,binding=0) coherent restrict buffer mks_uv_atlas_block { highp uint mks_atlas_words[%zu]; };\n"
        "highp vec2 mks_atlas_uv;\nhighp uint mks_atlas_sites=0u;\n"
        "// MKS_ATLAS_GLOBALS_END\n";
    const char *capture =
        "// MKS_ATLAS_SITE_BEGIN\n"
        "mks_atlas_uv=%s;\nmks_atlas_sites++;\n"
        "// MKS_ATLAS_SITE_END\n";
    const char *tail =
        "// MKS_ATLAS_TAIL_BEGIN\n"
        "ivec2 mks_atlas_p=ivec2(gl_FragCoord.xy);\n"
        "if(all(greaterThanEqual(mks_atlas_p,ivec2(0)))&&all(lessThan(mks_atlas_p,ivec2(%d,%d)))) {\n"
        "uint mks_atlas_i=uint(mks_atlas_p.y*%d+mks_atlas_p.x)*3u;\n"
        "uint mks_atlas_old=atomicAdd(mks_atlas_words[mks_atlas_i+2u],1u);\n"
        "if(mks_atlas_old==0u){mks_atlas_words[mks_atlas_i]=floatBitsToUint(mks_atlas_uv.x);"
        "mks_atlas_words[mks_atlas_i+1u]=floatBitsToUint(mks_atlas_uv.y);}\n"
        "if(mks_atlas_sites!=1u||any(isnan(mks_atlas_uv))||any(isinf(mks_atlas_uv)))"
        "atomicOr(mks_atlas_words[mks_atlas_i+2u],0x80000000u);\n}\n"
        "// MKS_ATLAS_TAIL_END\n";
    int g = snprintf(NULL, 0, globals, (size_t) width * height * 3);
    int c = snprintf(NULL, 0, capture, site.uv);
    int t = snprintf(NULL, 0, tail, width, height, width);
    if (g < 0 || c < 0 || t < 0 || len > SIZE_MAX - (size_t) g - c - t - 1)
        return NULL;
    size_t total = len + g + c + t + 1;
    char *out = malloc(total);
    if (!out) return NULL;
    size_t first = start - source, sample = first + site.start, pos = 0;
    memcpy(out, source, first); pos += first;
    snprintf(out + pos, total - pos, globals, (size_t) width * height * 3); pos += g;
    memcpy(out + pos, source + first, sample - first); pos += sample - first;
    snprintf(out + pos, total - pos, capture, site.uv); pos += c;
    memcpy(out + pos, source + sample, len - 1 - sample); pos += len - 1 - sample;
    snprintf(out + pos, total - pos, tail, width, height, width); pos += t;
    out[pos++] = '}'; out[pos] = 0;
    return out;
}

#endif
