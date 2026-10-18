#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <functional>

struct OctreeMeshCell {
    float sdf[8];
    glm::vec3 cubeMin;
    float cubeSize;
    int brushIndex;
};

struct OctreeMeshResult {
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> normals;
    uint32_t vertexCount = 0;
};

// App sets this when the GPU service is available; null in headless/server builds.
extern std::function<OctreeMeshResult(const std::vector<OctreeMeshCell>&, float)> gOctreeMeshCompute;
