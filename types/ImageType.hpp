#pragma once

#include <cstdint>

// Where a brush's texture comes from. The brush map (one entry per brush /
// material index) records the source so a single mesh can draw its layers from
// the base image set, the TextureMixer output set, or the video set -- and the
// renderer can pick the right array (and layer inside it) per brush.
//
// The three sources are kept apart ("isolated"):
//   * Image        - a base texture loaded from disk (texture-array layers).
//   * TextureMixer - a generated Perlin mix (texture-array layers, its own range).
//   * Video        - a decoded clip (the dedicated video array, its own slots).
enum class ImageType : uint32_t {
    Image = 0,        // base texture array layer
    TextureMixer = 1, // TextureMixer output (texture-array layer range)
    Video = 2,        // video-array slot
};

inline const char* imageTypeName(ImageType type) {
    switch (type) {
        case ImageType::TextureMixer: return "TextureMixer";
        case ImageType::Video: return "Video";
        default: return "Image";
    }
}

// Compatibility umbrella: consumers that include ImageType.hpp also get the
// brush-image records that use it.
#include "BrushImage.hpp"
#include "BrushMap.hpp"
