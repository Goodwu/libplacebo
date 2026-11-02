/* Default-off atlas consumer source transforms. No production shader uses this. */
#pragma once
#include "atlas_anchor.h"

static char *mks_read_replace(const char *s, size_t first, size_t last, const char *with)
{
    size_t n=strlen(s), w=strlen(with);
    if (first>last || last>n || n>1048576 || w>1048576) return NULL;
    char *r=malloc(n-(last-first)+w+1);
    if (!r) return NULL;
    memcpy(r,s,first); memcpy(r+first,with,w);
    memcpy(r+first+w,s+last,n-last+1);
    return r;
}

// B0 is created before probe insertion. There is no shader fallback or own-UV
// branch: its sole textureLod receives this one readonly atlas load directly.
static char *mks_atlas_b0(const char *source, const char *body, int w, int h)
{
    struct mks_sample_tap_site site;
    if (!mks_atlas_clean_a0(source) || !mks_sample_tap_find(body,&site) ||
        w<1 || h<1 || (size_t)w>4194304u/(size_t)h) return NULL;
    int version=0, used=0;
    if (sscanf(source,"#version %d es%n",&version,&used)!=1 || version<310 ||
        !used || source[used]!='\n') return NULL;
    const char *b=strstr(source,body), *out=strstr(source,"out vec4 out_color;\n");
    const char *main=strstr(source,"void main() {\nout_color = ");
    if (!b || strstr(b+strlen(body),body) || !out || !main || out>b || b>main ||
        strstr(out+1,"out vec4 out_color;\n")) return NULL;
    const char *p=main+strlen("void main() {\nout_color = "); char function[8];
    if (!mks_tap_ident(&p,function,NULL) || strcmp(p,"();\n}")) return NULL;
    char call[96];
    snprintf(call,sizeof(call),"textureLod(%s, mks_cache_uv(), 0.0)",site.tex);
    char *newbody=mks_read_replace(body,site.call_start,site.call_end,call);
    if (!newbody) return NULL;
    const char *fmt=
        "%.*s\n#extension GL_EXT_YUV_target : require\n%.*s"
        "layout(yuv) out highp vec3 out_color;\n%.*s"
        "layout(std430,binding=0) readonly restrict buffer mks_cache_block { highp uint mks_cache_words[%zu]; };\n"
        "highp vec2 mks_cache_uv() {\n"
        "uvec2 mks_cache_p=uvec2(gl_FragCoord.xy);\n"
        "uint mks_cache_i=(mks_cache_p.y*%uu+mks_cache_p.x)*3u;\n"
        "highp vec2 mks_cache_loaded=uintBitsToFloat(uvec2(mks_cache_words[mks_cache_i],mks_cache_words[mks_cache_i+1u]));\n"
        "return mks_cache_loaded;\n}\n%s%.*s"
        "highp float mks_half_store(highp float value) {\n"
        "highp uint raw=floatBitsToUint(value); highp uint mag=raw&0x7fffffffu;\n"
        "if(mag==0u||mag>=0x38800000u) return unpackHalf2x16(packHalf2x16(vec2(value,0.0))).x;\n"
        "highp uint sign=raw&0x80000000u; highp int e=int(mag>>23u)-127;\n"
        "if(e< -24) return uintBitsToFloat(sign);\n"
        "highp uint h=((mag&0x7fffffu)|0x800000u)>>uint(-e-1);\n"
        "highp float q=float(h)*0.000000059604644775390625;\n"
        "return uintBitsToFloat(floatBitsToUint(q)|sign); }\n"
        "void main() {\nhighp vec3 mks_rgb = %s().rgb;\n"
        "mks_rgb=vec3(mks_half_store(mks_rgb.r),mks_half_store(mks_rgb.g),mks_half_store(mks_rgb.b));\n"
        "highp float mks_y=0.2627*mks_rgb.r+0.6780*mks_rgb.g+0.0593*mks_rgb.b;\n"
        "highp float mks_cb=(mks_rgb.b-mks_y)*0.5/(1.0-0.0593);\n"
        "highp float mks_cr=(mks_rgb.r-mks_y)*0.5/(1.0-0.2627);\n"
        "out_color=vec3(mks_y,mks_cb+0.5,mks_cr+0.5);\n}";
    const char *oe=out+strlen("out vec4 out_color;\n"), *be=b+strlen(body);
    int n=snprintf(NULL,0,fmt,used,source,(int)(out-source-used-1),source+used+1,
        (int)(b-oe),oe,(size_t)w*h*3,(unsigned)w,newbody,(int)(main-be),be,function);
    char *r=n>0 ? malloc((size_t)n+1) : NULL;
    if (r) snprintf(r,(size_t)n+1,fmt,used,source,(int)(out-source-used-1),source+used+1,
        (int)(b-oe),oe,(size_t)w*h*3,(unsigned)w,newbody,(int)(main-be),be,function);
    free(newbody); return r;
}

// Exactly three insertions; removing marker spans reconstructs actual B0.
// UV and half use the same resulting program with only a runtime uniform change.
static char *mks_atlas_probe(const char *b0, int w, int h)
{
    if (!b0 || w<1 || h<1 || (size_t)w>4194304u/(size_t)h ||
        strstr(b0,"MKS_READ_") || strstr(b0,"mks_read_")) return NULL;
    const char *g=strstr(b0,"highp vec2 mks_cache_uv() {\n");
    const char *s=strstr(b0,"return mks_cache_loaded;\n");
    size_t len=strlen(b0);
    if (!g || !s || g>s || !len || b0[len-1]!='}' || b0[len-2]!='\n' ||
        strstr(g+1,"highp vec2 mks_cache_uv()") || strstr(s+1,"return mks_cache_loaded;")) return NULL;
    const char *fmt=
        "%.*s// MKS_READ_GLOBALS_BEGIN\n"
        "layout(std430,binding=1) coherent restrict buffer mks_read_block { highp uint mks_read_words[%zu]; };\n"
        "uniform highp int mks_read_mode;\nhighp vec2 mks_read_uv;\nhighp uint mks_read_sites=0u;\n"
        "// MKS_READ_GLOBALS_END\n%.*s"
        "// MKS_READ_SITE_BEGIN\nmks_read_uv=mks_cache_loaded;\nmks_read_sites++;\n// MKS_READ_SITE_END\n%.*s"
        "// MKS_READ_TAIL_BEGIN\n"
        "uvec2 mks_read_p=uvec2(gl_FragCoord.xy);\n"
        "uint mks_read_i=(mks_read_p.y*%uu+mks_read_p.x)*4u;\n"
        "uint mks_read_old=atomicAdd(mks_read_words[mks_read_i],1u);\n"
        "if(mks_read_old==0u) {\n"
        "if(mks_read_mode==0) {mks_read_words[mks_read_i+1u]=floatBitsToUint(mks_read_uv.x);"
        "mks_read_words[mks_read_i+2u]=floatBitsToUint(mks_read_uv.y);mks_read_words[mks_read_i+3u]=mks_read_sites;}\n"
        "else if(mks_read_mode==1) {mks_read_words[mks_read_i+1u]=floatBitsToUint(mks_rgb.r);"
        "mks_read_words[mks_read_i+2u]=floatBitsToUint(mks_rgb.g);mks_read_words[mks_read_i+3u]=floatBitsToUint(mks_rgb.b);}\n"
        "else {atomicOr(mks_read_words[mks_read_i],0x80000000u);}\n}\n"
        "if(mks_read_sites!=1u) atomicOr(mks_read_words[mks_read_i],0x80000000u);\n"
        "// MKS_READ_TAIL_END\n}";
    int n=snprintf(NULL,0,fmt,(int)(g-b0),b0,(size_t)w*h*4,(int)(s-g),g,
        (int)(b0+len-1-s),s,(unsigned)w);
    char *r=n>0 ? malloc((size_t)n+1) : NULL;
    if (r) snprintf(r,(size_t)n+1,fmt,(int)(g-b0),b0,(size_t)w*h*4,
        (int)(s-g),g,(int)(b0+len-1-s),s,(unsigned)w);
    return r;
}
