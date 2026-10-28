#pragma once

#include "../math/TextureCoordPainter.hpp"

// Single-axis world-position projection (GPU); data carries the scale.
class ProjectionZMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionZMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f));
    TextureType textureType() const override;
    glm::vec2 paintUV(const OctreeNode &o) const override;
private:
    glm::vec2 scale;
};
