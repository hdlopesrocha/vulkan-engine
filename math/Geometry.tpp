// Geometry.tpp - template implementations for InstanceGeometry

#ifndef GEOMETRY_TPP
#define GEOMETRY_TPP

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
    if(!geometry->reusable){
        delete geometry;
    }
}

#endif // GEOMETRY_TPP
