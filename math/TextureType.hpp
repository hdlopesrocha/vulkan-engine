#pragma once
#include <cstdint>

// Per-vertex texture-coordinate technique.
//
// The selected technique used to live on the material (TextureMappingMode); it
// is now painted per vertex by the TexturePainter (like brushIndex), so one
// mesh can mix mappings. The numbers stay in sync with the GLSL constants in
// shaders/includes/TextureMapping.glsl (TEXTURE_MAPPING_*) and with the
// material default (types/MaterialProperties.hpp aliases this enum).
//
// Triplanar is the default: its UVs are derived from the world position on the
// GPU, so a Triplanar vertex carries texCoord = (0, 0). The ProjectionX/Y/Z and
// Angular modes are likewise evaluated in the shader from the world position;
// only Custom consumes the painter's per-vertex UV.
enum class TextureType : uint32_t {
    Triplanar = 0,   // weighted X/Y/Z world-position projections (GPU)
    ProjectionX = 1, // single YZ plane projection (GPU)
    ProjectionY = 2, // single XZ plane projection (GPU)
    ProjectionZ = 3, // single XY plane projection (GPU)
    Angular = 4,     // spherical/angular coordinates around mappingOrigin (GPU)
    Custom = 5,      // painter-authored per-vertex UVs
};

inline const char* textureTypeName(TextureType type) {
    switch (type) {
        case TextureType::ProjectionX: return "Projection X";
        case TextureType::ProjectionY: return "Projection Y";
        case TextureType::ProjectionZ: return "Projection Z";
        case TextureType::Angular: return "Angular";
        case TextureType::Custom: return "Custom (UV)";
        default: return "Triplanar";
    }
}
