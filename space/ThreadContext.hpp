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
