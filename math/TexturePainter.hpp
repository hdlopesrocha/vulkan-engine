#pragma once
#include "Vertex.hpp"
#include <glm/glm.hpp>

class TexturePainter {
public:
    virtual int paint(const Vertex &v, glm::vec4 translate, glm::vec4 scale) const = 0;
};

 
