#pragma once

#include <functional>

#include "../math/TextureCoordPainter.hpp"

// The UV is authored here (a caller-supplied function over the node); the
// shader samples it directly.
class CustomMappingPainter : public TextureCoordPainter {
public:
    CustomMappingPainter();
    explicit CustomMappingPainter(std::function<glm::vec2(const OctreeNode &)> fn);

    TextureType textureType() const override;
    glm::vec2 paintUV(const OctreeNode &o) const override;

private:
    std::function<glm::vec2(const OctreeNode &)> uvFn;
};
