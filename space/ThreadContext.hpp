#pragma once

#include <glm/glm.hpp>
#include <type_traits>
#if __has_include(<tsl/robin_map.h>)
#include <tsl/robin_map.h>
#else
#include <unordered_map>
namespace tsl {
    template <typename K, typename V, typename H = std::hash<K>>
    using robin_map = std::unordered_map<K, V, H>;
}
#endif
// Prefer tsl::robin_set when available, otherwise fall back to std::unordered_set
#if __has_include(<tsl/robin_set.h>)
#include <tsl/robin_set.h>
#else
#include <unordered_set>
namespace tsl {
    template <typename K, typename H = std::hash<K>>
    using robin_set = std::unordered_set<K, H>;
}
#endif
#include <shared_mutex>
#include "../math/BoundingCube.hpp"
#include "OctreeNodeLevel.hpp"

class ThreadContext {
public:
    tsl::robin_map<glm::vec3, float> shapeSdfCache;
    tsl::robin_map<glm::vec4, OctreeNodeLevel> nodeCache;
    // Tracks which shifted cubes have already been passed to the border handler
    // to avoid calling the handler multiple times for the same cube+level.
    tsl::robin_set<glm::vec4> invokedCubeCalls;
    std::shared_mutex mutex;
    BoundingCube cube;
    ThreadContext(BoundingCube cube);
};
