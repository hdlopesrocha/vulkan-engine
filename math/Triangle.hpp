// Auto-generated wrapper header for Triangle
#pragma once

#include "Common.hpp"

struct Triangle {
public:
    Vertex v[3];
    Triangle(Vertex v1, Vertex v2, Vertex v3){ v[0]=v1; v[1]=v2; v[2]=v3; }
    Triangle flip(){ Vertex t=v[1]; v[1]=v[2]; v[2]=t; return *this; }
};
