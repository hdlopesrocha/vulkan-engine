#pragma once
#include "Geometry.hpp"
#include <vector>

template <typename T>
struct InstanceGeometry {
public:
    Geometry * geometry;
    std::vector<T> instances;
    InstanceGeometry(Geometry * geom);
    InstanceGeometry(Geometry * geom, std::vector<T> &insts);
    ~InstanceGeometry();
};

#include "Geometry.tpp"
