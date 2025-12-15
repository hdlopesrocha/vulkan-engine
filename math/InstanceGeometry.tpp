#pragma once
#include "Geometry.hpp"
#include <vector>

template <typename T>
struct InstanceGeometry {
public:
    Geometry * geometry;
    std::vector<T> instances;
    InstanceGeometry(Geometry * geometry);
    InstanceGeometry(Geometry * geometry, std::vector<T> &instances);
    ~InstanceGeometry();
};

#include "Geometry.tpp"
