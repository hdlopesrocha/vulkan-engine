#include "TextureCoordPainter.hpp"

TextureCoordPainter::~TextureCoordPainter() = default;

glm::vec2 TextureCoordPainter::paintUV(const OctreeNode &o) const {
    (void)o;
    return glm::vec2(0.0f);
}

bool TextureCoordPainter::needsPosition() const {
    return false;
}
