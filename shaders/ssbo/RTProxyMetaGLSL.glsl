#ifndef R_T_PROXY_META_G_L_S_L_GLSL
#define R_T_PROXY_META_G_L_S_L_GLSL

// Extracted from shaders/includes/rt/RtParams.glsl (single-struct GLSL type).

struct RTProxyMetaGLSL {    vec4 minAndMatId;  // xyz=AABB min, w=material id
    vec4 maxAndFlags;  // xyz=AABB max, w=flags
    vec4 albedoRough;  // rgb=avg albedo, a=roughness
    vec4 extra;        // x=horizontal footprint (max x/z extent, for coarse-box fallback), yzw reserved
};

#endif // R_T_PROXY_META_G_L_S_L_GLSL
