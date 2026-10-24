#pragma once

#include <string>

// One source texture set (albedo/normal/height/roughness/AO), in the engine's
// fixed layer order.
struct SceneTextureTriple {
    std::string albedo;
    std::string normal;
    std::string height;
    std::string rough;
    std::string ao;
};
