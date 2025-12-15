#pragma once
#include "AbstractBoundingBox.hpp"
#include <glm/glm.hpp>

class Frustum {
public:
    Frustum() {}
    Frustum(glm::mat4 m);
    ContainmentType test(const AbstractBoundingBox &box);
private:
    enum Planes { Left = 0, Right, Bottom, Top, Near, Far, Count, Combinations = Count * (Count - 1) / 2 };
    template<Planes i, Planes j> struct ij2k { enum { k = i * (9 - i) / 2 + j - 1 }; };
    template<Planes a, Planes b, Planes c> glm::vec3 intersection(glm::vec3* crosses);
    glm::vec4   m_planes[Count];
    glm::vec3   m_points[8];
};

 
