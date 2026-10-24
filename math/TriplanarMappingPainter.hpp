#pragma once

#include "TextureCoordPainter.hpp"

// Weighted X/Y/Z world-position projections computed on the GPU; data carries
// the (scaleU, scaleV) applied to the projection.
class TriplanarMappingPainter : public TextureCoordPainter {
public:
    explicit TriplanarMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::Triplanar; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};
