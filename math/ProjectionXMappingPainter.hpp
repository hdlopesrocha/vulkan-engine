#pragma once

#include "TextureCoordPainter.hpp"

// Single-axis world-position projection (GPU); data carries the scale.
class ProjectionXMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionXMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionX; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};
