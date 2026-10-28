#include "CustomMappingPainter.hpp"

#include <utility>

CustomMappingPainter::CustomMappingPainter() = default;

CustomMappingPainter::CustomMappingPainter(std::function<glm::vec2(const OctreeNode &)> fn)
    : uvFn(std::move(fn)) {}

TextureType CustomMappingPainter::textureType() const {
    return TextureType::Custom;
}

glm::vec2 CustomMappingPainter::paintUV(const OctreeNode &o) const {
    return uvFn ? uvFn(o) : glm::vec2(0.0f);
}
