#ifndef SDF_MATERIAL_TYPE_GLSL
#define SDF_MATERIAL_TYPE_GLSL

// Material shading mode encoded in SdfMaterial::surfaceParams.w (float).
// CPU twin: sdf/types/SdfMaterialType.hpp (SdfMaterialType).
// The render mode actually used by the march comes from the params UBO;
// these ids document/validate the material payload.
#define SDF_MAT_SURFACE 0u
#define SDF_MAT_EMISSIVE 1u
#define SDF_MAT_VOLUME 2u
#define SDF_MAT_TRANSPARENT 3u

#endif // SDF_MATERIAL_TYPE_GLSL
