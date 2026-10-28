#include "WaterBrush.hpp"

WaterBrush::WaterBrush(int water_){
    this->water = water_;
}

int WaterBrush::paint(const OctreeNode &o) const {
    (void)o;
    return water;
}

