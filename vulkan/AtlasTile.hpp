#pragma once

#include <string>
// Represents a single tile/region in a texture atlas
struct AtlasTile {
    std::string name;
    float offsetX = 0.0f;      // UV offset X (0-1)
    float offsetY = 0.0f;      // UV offset Y (0-1)
    float scaleX = 1.0f;       // UV scale X (0-1)
    float scaleY = 1.0f;       // UV scale Y (0-1)
};