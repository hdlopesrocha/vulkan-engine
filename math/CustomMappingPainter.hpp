#pragma once

#include <functional>
#include <utility>

#include "TextureCoordPainter.hpp"

// The UV is authored here (a caller-supplied function over the node); the
// shader samples it directly.
class CustomMappingPainter : public TextureCoordPainter {
public:
    CustomMappingPainter() = default;
    explicit CustomMappingPainter(std::function<glm::vec2(const OctreeNode &)> fn) : uvFn(std::move(fn)) {}

    TextureType textureType() const override { return TextureType::Custom; }
    glm::vec2 paintUV(const OctreeNode &o) const override { return uvFn ? uvFn(o) : glm::vec2(0.0f); }

private:
    std::function<glm::vec2(const OctreeNode &)> uvFn;
};
