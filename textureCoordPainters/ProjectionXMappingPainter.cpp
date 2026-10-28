#include "ProjectionXMappingPainter.hpp"

ProjectionXMappingPainter::ProjectionXMappingPainter(glm::vec2 uvScale) : scale(uvScale) {}

TextureType ProjectionXMappingPainter::textureType() const {
    return TextureType::ProjectionX;
}

glm::vec2 ProjectionXMappingPainter::paintUV(const OctreeNode &) const {
    return scale;
}
