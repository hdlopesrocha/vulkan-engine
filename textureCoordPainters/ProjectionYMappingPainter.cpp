#include "ProjectionYMappingPainter.hpp"

ProjectionYMappingPainter::ProjectionYMappingPainter(glm::vec2 uvScale) : scale(uvScale) {}

TextureType ProjectionYMappingPainter::textureType() const {
    return TextureType::ProjectionY;
}

glm::vec2 ProjectionYMappingPainter::paintUV(const OctreeNode &) const {
    return scale;
}
