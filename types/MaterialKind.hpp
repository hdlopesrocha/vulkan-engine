#pragma once

#include <cstdint>

// What a material samples: a layer of the scene texture arrays, one slot of the
// video texture array, or the water params palette. Materials of kind Video are
// shaded by the video layer renderer (rendered after solid and water so the
// per-material blend mode can be applied against the composed scene color).
enum class MaterialKind : uint32_t {
    Texture = 0,
    Video = 1,
    Water = 2,
};

inline const char* materialKindName(MaterialKind kind) {
    switch (kind) {
        case MaterialKind::Video: return "Video";
        case MaterialKind::Water: return "Water";
        default: return "Textures";
    }
}
