#pragma once

#include "../math/TextureCoordPainter.hpp"

// Weighted X/Y/Z world-position projections computed on the GPU; data carries
// the (scaleU, scaleV) applied to the projection.
class TriplanarMappingPainter : public TextureCoordPainter {
public:
    explicit TriplanarMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f));
    TextureType textureType() const override;
    glm::vec2 paintUV(const OctreeNode &o) const override;
private:
    glm::vec2 scale;
};
