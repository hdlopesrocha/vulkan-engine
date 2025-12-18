#pragma once
#include "Geometry.hpp"

template <typename T>
InstanceGeometry<T>::InstanceGeometry(Geometry * geometry) {
    this->geometry = geometry;
    geometry->setCenter();
}

template <typename T>
InstanceGeometry<T>::InstanceGeometry(Geometry * geometry, std::vector<T> &instances) {
    this->geometry = geometry;
    this->instances = instances;
    geometry->setCenter();
}

template <typename T>
InstanceGeometry<T>::~InstanceGeometry() {

}

