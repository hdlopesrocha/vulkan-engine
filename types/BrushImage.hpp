#pragma once

#include <cstdint>

#include "ImageType.hpp"

// One brush's image: the source category plus the index inside that category.
// For Image/TextureMixer `index` is the texture-array layer (the two ranges live
// in the same 5-map array); for Video it is the video-array slot.
struct BrushImage {
    ImageType type = ImageType::Image;
    uint32_t index = 0;
};
