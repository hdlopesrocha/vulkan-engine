#pragma once

#include <cstddef>
#include <vector>

#include "BrushImage.hpp"

// The per-brush map: brushMap[i] is the image brush index i samples. Uploaded to
// the GPU as a (type, index) table the shaders resolve from.
struct BrushMap {
    std::vector<BrushImage> entries;

    void resize(size_t count) { entries.assign(count, BrushImage{}); }
    BrushImage& operator[](size_t i) { return entries[i]; }
    const BrushImage& operator[](size_t i) const { return entries[i]; }
    size_t size() const { return entries.size(); }
};
