// Auto-generated wrapper header for ThreadContext
#pragma once

#include <glm/glm.hpp>
#include <tsl/robin_map.h>
#include <shared_mutex>
#include "../math/BoundingCube.hpp"
#include "OctreeNodeLevel.hpp"

class ThreadContext {
public:
    tsl::robin_map<glm::vec3, float> shapeSdfCache;
    tsl::robin_map<glm::vec4, OctreeNodeLevel> nodeCache;
    std::shared_mutex mutex;
    BoundingCube cube;
    ThreadContext(BoundingCube cube);
};
