#pragma once
#include "AbstractBoundingBox.hpp"
#include <glm/glm.hpp>

class Frustum {
public:
    Frustum() {}
    Frustum(glm::mat4 m);
    ContainmentType test(const AbstractBoundingBox &box);
private:
    enum Planes { Left = 0, Right, Bottom, Top, Near, Far, Count };
    glm::vec4   m_planes[Count];
};

 
