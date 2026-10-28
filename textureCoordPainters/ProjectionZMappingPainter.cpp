#include "ProjectionZMappingPainter.hpp"

ProjectionZMappingPainter::ProjectionZMappingPainter(glm::vec2 uvScale) : scale(uvScale) {}

TextureType ProjectionZMappingPainter::textureType() const {
    return TextureType::ProjectionZ;
}

glm::vec2 ProjectionZMappingPainter::paintUV(const OctreeNode &) const {
    return scale;
}
