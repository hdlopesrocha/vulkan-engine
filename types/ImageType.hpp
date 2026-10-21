#pragma once
#include <cstdint>
#include <vector>

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

// One brush's image: the source category plus the index inside that category.
// For Image/TextureMixer `index` is the texture-array layer (the two ranges live
// in the same 5-map array); for Video it is the video-array slot.
struct BrushImage {
    ImageType type = ImageType::Image;
    uint32_t index = 0;
};

inline const char* imageTypeName(ImageType type) {
    switch (type) {
        case ImageType::TextureMixer: return "TextureMixer";
        case ImageType::Video: return "Video";
        default: return "Image";
    }
}

// The per-brush map: brushMap[i] is the image brush index i samples. Uploaded to
// the GPU as a (type, index) table the shaders resolve from.
struct BrushMap {
    std::vector<BrushImage> entries;

    void resize(size_t count) { entries.assign(count, BrushImage{}); }
    BrushImage& operator[](size_t i) { return entries[i]; }
    const BrushImage& operator[](size_t i) const { return entries[i]; }
    size_t size() const { return entries.size(); }
};
