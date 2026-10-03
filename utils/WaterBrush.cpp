#include "WaterBrush.hpp"

WaterBrush::WaterBrush(int water_){
    this->water = water_;
}

int WaterBrush::paint(const Vertex &vertex) const {
    (void)vertex;
    return water;
}

