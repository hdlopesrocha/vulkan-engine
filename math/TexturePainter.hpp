#pragma once
#include "Vertex.hpp"
#include <glm/glm.hpp>

class TexturePainter {
public:
    virtual int paint(const Vertex &v) const = 0;
    virtual glm::vec3 paintHSV(const Vertex &v) const { return glm::vec3(0.0f, 0.0f, 1.0f); }
};

 
