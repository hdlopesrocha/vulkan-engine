#pragma once
#include <string>
#include <vector>
#include "BillboardLayer.hpp"

// Represents a complete billboard composed of multiple layers
struct Billboard {
    std::string name;
    std::vector<BillboardLayer> layers;
    
    // Billboard metadata
    float width = 1.0f;        // Physical width in world units
    float height = 1.0f;       // Physical height in world units
};