#pragma once

#include "../math/TextureCoordPainter.hpp"

// Spherical coordinates around the material's mapping origin (GPU); tiling
// comes from mappingOrigin.w, so data carries (0, 0).
class AngularMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override;
    glm::vec2 paintUV(const OctreeNode &o) const override;
};
