#pragma once
#include "Geometry.hpp"

class SphereGeometry : public Geometry {
    int lats;
    int longs;
public:
    SphereGeometry(int lats, int longs);
    void addTriangle(glm::vec3 a,glm::vec3 b, glm::vec3 c);
};

 
