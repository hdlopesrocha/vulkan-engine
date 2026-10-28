#include "TriplanarMappingPainter.hpp"

TriplanarMappingPainter::TriplanarMappingPainter(glm::vec2 uvScale) : scale(uvScale) {}

TextureType TriplanarMappingPainter::textureType() const {
    return TextureType::Triplanar;
}

glm::vec2 TriplanarMappingPainter::paintUV(const OctreeNode &) const {
    return scale;
}
