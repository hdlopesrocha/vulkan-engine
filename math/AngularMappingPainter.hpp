#pragma once

#include "TextureCoordPainter.hpp"

// Spherical coordinates around the material's mapping origin (GPU); tiling
// comes from mappingOrigin.w, so data carries (0, 0).
class AngularMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::Angular; }
    glm::vec2 paintUV(const OctreeNode &) const override { return glm::vec2(0.0f); }
};
