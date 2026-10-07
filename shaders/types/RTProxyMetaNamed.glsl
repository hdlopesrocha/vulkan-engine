#ifndef R_T_PROXY_META_NAMED_GLSL
#define R_T_PROXY_META_NAMED_GLSL

// Extracted from shaders/includes/rt/RtParams.glsl (single-struct GLSL type).

// Named view over the packed RTProxyMetaGLSL - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct RTProxyMetaNamed {
    vec3 boxMin;
    vec2 boxMinXZ;   // horizontal extent of boxMin
    float materialId;
    vec3 boxMax;
    vec2 boxMaxXZ;   // horizontal extent of boxMax
    float boxTop;    // boxMax.y: the box top edge
    bool isWater;
    vec3 albedo;
    float roughness;
    float footprint;
};

#endif // R_T_PROXY_META_NAMED_GLSL
