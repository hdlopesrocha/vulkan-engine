#include "AngularMappingPainter.hpp"

TextureType AngularMappingPainter::textureType() const {
    return TextureType::Angular;
}

glm::vec2 AngularMappingPainter::paintUV(const OctreeNode &) const {
    return glm::vec2(0.0f);
}
