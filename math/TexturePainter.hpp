#pragma once
#include "Vertex.hpp"
#include <glm/glm.hpp>

class TexturePainter {
public:
    virtual int paint(const Vertex &v) const = 0;
};

 
