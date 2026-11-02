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

/* Strict diagnostic transform of the one supported generated sampling site.
 * No ordinary shader generation uses this helper. Unknown syntax fails closed. */
#pragma once

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mks_sample_tap_site {
    unsigned tex_id, uv_id;
    size_t start, end;
    char scale[8], tex[8], uv[8];
    size_t call_start, call_end;
};

static const char *mks_tap_space(const char *p)
{
    while (*p && isspace((unsigned char) *p))
        p++;
    return p;
}

static bool mks_tap_token(const char **p, const char *token)
{
    *p = mks_tap_space(*p);
    size_t len = strlen(token);
    if (strncmp(*p, token, len) ||
        ((isalnum((unsigned char) token[len-1]) || token[len-1] == '_') &&
         (isalnum((unsigned char) (*p)[len]) || (*p)[len] == '_')))
        return false;
    *p += len;
    return true;
}

static bool mks_tap_ident(const char **p, char out[8], unsigned *id)
{
    *p = mks_tap_space(*p);
    const char *begin = *p;
    if (*(*p)++ != '_')
        return false;
    unsigned value = 0, digits = 0;
    while (**p && strchr("0123456789abcdef", **p)) {
        if (++digits > 4)
            return false;
        unsigned c = (unsigned char) *(*p)++;
        value = value * 16 + (c <= '9' ? c - '0' : c - 'a' + 10);
    }
    if (!digits || isalnum((unsigned char) **p) || **p == '_')
        return false;
    size_t len = *p - begin;
    memcpy(out, begin, len);
    out[len] = '\0';
    if (id)
        *id = value;
    return true;
}

static bool mks_sample_tap_find(const char *body, struct mks_sample_tap_site *out)
{
    const char *marker = "// pl_shader_sample_direct";
    if (!body || !out || strstr(body, "mks_tap_") || strstr(body, "mks_sample_value"))
        return false;
    const char *site = strstr(body, marker);
    if (!site || strstr(site + strlen(marker), marker))
        return false;
    const char *p = site + strlen(marker);
    while (*p == ' ' || *p == '\t' || *p == '\r')
        p++;
    if (*p++ != '\n')
        return false;
    struct mks_sample_tap_site s = {0};
    p = mks_tap_space(p);
    s.start = p - body;
    if (!mks_tap_token(&p, "vec4") || !mks_tap_token(&p, "color") ||
        !mks_tap_token(&p, "=") || !mks_tap_token(&p, "vec4") ||
        !mks_tap_token(&p, "(") || !mks_tap_ident(&p, s.scale, NULL) ||
        !mks_tap_token(&p, ")") || !mks_tap_token(&p, "*") ||
        !mks_tap_token(&p, "vec4") || !mks_tap_token(&p, "("))
        return false;
    p = mks_tap_space(p);
    s.call_start = p - body;
    if (!mks_tap_token(&p, "textureLod") || !mks_tap_token(&p, "(") ||
        !mks_tap_ident(&p, s.tex, &s.tex_id) || !mks_tap_token(&p, ",") ||
        !mks_tap_ident(&p, s.uv, &s.uv_id) || !mks_tap_token(&p, ",") ||
        !mks_tap_token(&p, "0.0") || !mks_tap_token(&p, ")"))
        return false;
    s.call_end = p - body;
    if (!mks_tap_token(&p, ")") || !mks_tap_token(&p, ";"))
        return false;
    s.end = p - body;
    // Count every texture-family invocation, even inside comments. False
    // negatives are safe here: only exactly one supported site is accepted.
    unsigned samples = 0;
    for (const char *scan = body; *scan;) {
        if (!(isalpha((unsigned char) *scan) || *scan == '_')) {
            scan++;
            continue;
        }
        const char *word = scan++;
        while (isalnum((unsigned char) *scan) || *scan == '_')
            scan++;
        size_t len = scan - word;
        bool sampling = (len >= 7 && !strncmp(word, "texture", 7)) ||
                        (len >= 10 && !strncmp(word, "texelFetch", 10));
        if (sampling && *mks_tap_space(scan) == '(') {
            if (++samples != 1 || word != body + s.call_start ||
                len != strlen("textureLod"))
                return false;
        }
    }
    if (samples != 1)
        return false;
    *out = s;
    return true;
}

// malloc-owned result; the original body remains untouched. The sampled
// expression and its tex/UV/LOD arguments are copied byte for byte exactly once.
static char *mks_sample_tap_instrument(const char *body,
                                       const struct mks_sample_tap_site *s)
{
    const char *globals = "highp vec2 mks_tap_uv;\nhighp vec4 mks_tap_sample;\n"
                          "highp uint mks_tap_count;\n";
    const char *fmt = "mks_tap_uv=%s;\nhighp vec4 mks_sample_value=vec4(%.*s);\n"
                      "mks_tap_sample=mks_sample_value;\nmks_tap_count++;\n"
                      "vec4 color=vec4(%s)*mks_sample_value;";
    size_t len = strlen(body), call = s->call_end - s->call_start;
    if (s->start > s->end || s->end > len || s->call_start > s->call_end ||
        s->call_end > s->end || call > 4096)
        return NULL;
    int added = snprintf(NULL, 0, fmt, s->uv, (int) call, body + s->call_start, s->scale);
    if (added < 0)
        return NULL;
    size_t total = strlen(globals) + s->start + (size_t) added + len - s->end + 1;
    char *out = malloc(total);
    if (!out)
        return NULL;
    size_t pos = strlen(globals);
    memcpy(out, globals, pos);
    memcpy(out + pos, body, s->start);
    pos += s->start;
    snprintf(out + pos, total - pos, fmt, s->uv, (int) call, body + s->call_start, s->scale);
    pos += added;
    memcpy(out + pos, body + s->end, len - s->end + 1);
    return out;
}

// B-only UV source intervention. A retains mks_sample_tap_instrument verbatim.
// Both runtime modes read/check the same A words and have identical source.
// Only one textureLod is emitted; no CPU UV reconstruction or second sampling.
static char *mks_uv_oracle_instrument(const char *body,
    const struct mks_sample_tap_site *s, const char *words,
    int x, int y, int width, int height)
{
    if (!body || !s || !words || !*words || x < 0 || y < 0 ||
        width < 1 || width > 256 || height < 1 || height > 256 ||
        strstr(body, "mks_oracle_") || strstr(body, "mks_uv_oracle_") ||
        s->start > s->end || s->end > strlen(body))
        return NULL;
    const char *head =
        "highp vec2 mks_tap_uv;\nhighp vec4 mks_tap_sample;\n"
        "highp uint mks_tap_count;\nuniform highp int mks_uv_oracle_use_a;\n"
        "bool mks_oracle_valid;\n"
        "highp vec2 mks_oracle_uv(highp vec2 own_uv) {\n"
        "mks_oracle_valid=true;\n"
        "ivec2 p=ivec2(gl_FragCoord.xy)-ivec2(%d,%d);\n"
        "if(all(greaterThanEqual(p,ivec2(0)))&&all(lessThan(p,ivec2(%d,%d)))) {\n"
        "uint i=uint(p.y*%d+p.x)*26u;\n"
        "if(%s[i+10u]!=1u||%s[i+25u]!=1u) {mks_oracle_valid=false; return own_uv;}\n"
        "highp vec2 a_uv=vec2(uintBitsToFloat(%s[i+20u]),uintBitsToFloat(%s[i+21u]));\n"
        "if(any(isnan(a_uv))||any(isinf(a_uv))) {mks_oracle_valid=false; return own_uv;}\n"
        "return mks_uv_oracle_use_a==1 ? a_uv : own_uv;\n}\nreturn own_uv;\n}\n";
    const char *statement =
        "mks_tap_uv=mks_oracle_uv(%s);\n"
        "highp vec4 mks_sample_value=vec4(textureLod(%s, mks_tap_uv, 0.0));\n"
        "mks_tap_sample=mks_sample_value;\n"
        "mks_tap_count+=mks_oracle_valid ? 1u : 0u;\n"
        "vec4 color=vec4(%s)*mks_sample_value;";
    int h = snprintf(NULL, 0, head, x, y, width, height, width, words, words, words, words);
    int n = snprintf(NULL, 0, statement, s->uv, s->tex, s->scale);
    if (h < 0 || n < 0)
        return NULL;
    size_t len = strlen(body), total = h + s->start + n + len - s->end + 1;
    char *out = malloc(total);
    if (!out)
        return NULL;
    snprintf(out, total, head, x, y, width, height, width, words, words, words, words);
    size_t pos = h;
    memcpy(out + pos, body, s->start);
    pos += s->start;
    snprintf(out + pos, total - pos, statement, s->uv, s->tex, s->scale);
    pos += n;
    memcpy(out + pos, body + s->end, len - s->end + 1);
    return out;
}

// Independent B-only probe: preserve A0 and all original 26-word semantics.
// Observed A fields are fetched once per helper call and the same values drive
// the guard, selected UV, and sidecar. Counts/UV are not an atomic snapshot.
static char *mks_uv_oracle_probe_instrument(const char *body,
    const struct mks_sample_tap_site *s, const char *words,
    int x, int y, int width, int height, int binding, unsigned int repetitions)
{
    if (!body || !s || !words || !*words || x < 0 || y < 0 || binding < 0 ||
        (repetitions != 1 && repetitions != 2) ||
        width < 1 || width > 256 || height < 1 || height > 256 ||
        strstr(body, "mks_oracle_") || strstr(body, "mks_uv_oracle_") ||
        strstr(body, "mks_probe_") || s->start > s->end || s->end > strlen(body))
        return NULL;
    const char *head =
        "layout(std430,binding=%d) coherent restrict buffer mks_oracle_probe { highp uint mks_probe_words[%zu]; };\n"
        "uniform highp int mks_probe_pair;\n"
        "highp uint mks_probe_helpers=0u;\nhighp uint mks_probe_sites=0u;\n"
        "highp uvec2 mks_probe_seen_counts=uvec2(0u);\n"
        "highp uvec2 mks_probe_seen_uv=uvec2(0u);\n"
        "highp uint mks_probe_reason=1u;\nhighp uint mks_probe_selected=2u;\n"
        "highp vec2 mks_tap_uv;\nhighp vec4 mks_tap_sample;\n"
        "highp uint mks_tap_count;\nuniform highp int mks_uv_oracle_use_a;\n"
        "bool mks_oracle_valid;\n"
        "highp vec2 mks_oracle_uv(highp vec2 own_uv) {\n"
        "mks_probe_helpers++;\nmks_oracle_valid=true;\n"
        "ivec2 p=ivec2(gl_FragCoord.xy)-ivec2(%d,%d);\n"
        "if(all(greaterThanEqual(p,ivec2(0)))&&all(lessThan(p,ivec2(%d,%d)))) {\n"
        "uint i=uint(p.y*%d+p.x)*26u;\n"
        "mks_probe_seen_counts=uvec2(%s[i+10u],%s[i+25u]);\n"
        "mks_probe_seen_uv=uvec2(%s[i+20u],%s[i+21u]);\n"
        "highp vec2 a_uv=uintBitsToFloat(mks_probe_seen_uv);\n"
        "mks_probe_reason=0u;\n"
        "if(mks_probe_seen_counts.x!=1u) mks_probe_reason|=2u;\n"
        "if(mks_probe_seen_counts.y!=1u) mks_probe_reason|=4u;\n"
        "if(any(isnan(a_uv))||any(isinf(a_uv))) mks_probe_reason|=8u;\n"
        "if(mks_probe_reason!=0u) {mks_oracle_valid=false; return own_uv;}\n"
        "mks_probe_selected=mks_uv_oracle_use_a==1 ? 1u : 0u;\n"
        "return mks_uv_oracle_use_a==1 ? a_uv : own_uv;\n}\nmks_oracle_valid=false;\nreturn own_uv;\n}\n";
    const char *statement =
        "mks_tap_uv=mks_oracle_uv(%s);\n"
        "highp vec4 mks_sample_value=vec4(textureLod(%s, mks_tap_uv, 0.0));\n"
        "mks_probe_sites++;\nmks_tap_sample=mks_sample_value;\n"
        "mks_tap_count+=mks_oracle_valid ? 1u : 0u;\n"
        "vec4 color=vec4(%s)*mks_sample_value;";
    size_t size = (size_t) width * height * 10 * repetitions;
    int h = snprintf(NULL, 0, head, binding, size, x, y, width, height, width,
                     words, words, words, words);
    int n = snprintf(NULL, 0, statement, s->uv, s->tex, s->scale);
    if (h < 0 || n < 0)
        return NULL;
    size_t len = strlen(body), total = h + s->start + n + len - s->end + 1;
    char *out = malloc(total);
    if (!out)
        return NULL;
    snprintf(out, total, head, binding, size, x, y, width, height, width,
             words, words, words, words);
    size_t pos = h;
    memcpy(out + pos, body, s->start);
    pos += s->start;
    snprintf(out + pos, total - pos, statement, s->uv, s->tex, s->scale);
    pos += n;
    memcpy(out + pos, body + s->end, len - s->end + 1);
    return out;
}

// Exact B-only experiment: retain A0 byte for byte and insert two declarations
// before the generated main. Reject unsupported source rather than broadening
// the transform or guessing a varying identifier.
static char *mks_invariant_b_vertex(const char *source, const char *varying)
{
    if (!source || !varying || strstr(source, "invariant"))
        return NULL;
    const char *id_end = varying;
    char id[8];
    unsigned value = 0;
    if (!mks_tap_ident(&id_end, id, &value) || *id_end || !value)
        return NULL;
    const char *main = strstr(source, "void main() {");
    if (!main || strstr(main + 1, "void main() {") ||
        (main != source && main[-1] != '\n'))
        return NULL;
    char decl[64];
    snprintf(decl, sizeof(decl), "out vec2 %s;\n", id);
    const char *out = strstr(source, decl);
    if (!out || out >= main || (out != source && out[-1] != '\n') ||
        strstr(out + 1, decl))
        return NULL;
    // The generated pass-through varying occurs only at its declaration and
    // assignment. Extra uses/declarations (including differently spaced ones)
    // are outside this narrowly supported experiment.
    unsigned uses = 0;
    size_t id_len = strlen(id);
    for (const char *scan = source; (scan = strstr(scan, id)); scan += id_len) {
        bool left = scan != source && (isalnum((unsigned char) scan[-1]) || scan[-1] == '_');
        bool right = isalnum((unsigned char) scan[id_len]) || scan[id_len] == '_';
        if (!left && !right)
            uses++;
    }
    char assignment[32];
    snprintf(assignment, sizeof(assignment), "%s = ", id);
    const char *write = strstr(main, assignment);
    if (uses != 2 || !write || strstr(write + 1, assignment))
        return NULL;
    const char *position = strstr(source, "gl_Position = vec4(va_pos, 0.0, 1.0);");
    const char *first_position = strstr(source, "gl_Position");
    if (!position || position < main || first_position != position ||
        strstr(position + strlen("gl_Position"), "gl_Position"))
        return NULL;
    char added[96];
    int extra = snprintf(added, sizeof(added), "invariant gl_Position;\ninvariant %s;\n", id);
    if (extra < 0 || (size_t) extra >= sizeof(added))
        return NULL;
    size_t len = strlen(source), prefix = main - source;
    if (len > (size_t) -1 - (size_t) extra - 1)
        return NULL;
    char *result = malloc(len + extra + 1);
    if (!result)
        return NULL;
    memcpy(result, source, prefix);
    memcpy(result + prefix, added, extra);
    memcpy(result + prefix + extra, main, len - prefix + 1);
    return result;
}
