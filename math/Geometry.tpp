#pragma once
#include "Geometry.hpp"

template <typename T>
InstanceGeometry<T>::InstanceGeometry(Geometry * geom) {
    this->geometry = geom;
    geom->setCenter();
}

template <typename T>
InstanceGeometry<T>::InstanceGeometry(Geometry * geom, std::vector<T> &insts) {
    this->geometry = geom;
    this->instances = insts;
    geom->setCenter();
}

template <typename T>
InstanceGeometry<T>::~InstanceGeometry() {

}

