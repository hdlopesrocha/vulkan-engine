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
#include "../math/geometry/BoundingCube.hpp"
#include "OctreeNodeLevel.hpp"
#include "ShapeSdfCache.hpp"

class OctreeThreadContext {
public:
    // Shape-SDF value cache (integer lattice keys, generation-tagged). See
    // ShapeSdfCache for the exactness and lifetime rules.
    ShapeSdfCache shapeSdfCache;
    tsl::robin_map<glm::vec4, OctreeNodeLevel> nodeCache;
    // child node -> (parent node, child index within parent). Populated by the
    // tree traversal (IteratorHandler), seeded with the world-root path to the
    // chunk root, so neighbor lookups during triangle extraction can rebuild
    // root-consistent cubes (see OctreeTriangleTessellator::iterateTriangles in
    // space/OctreeTriangleTessellator.hpp). The parent link is
    // NOT stored inside the octree nodes.
    tsl::robin_map<OctreeNode*, std::pair<OctreeNode*, int>> parentOf;
    OctreeThreadContext();
};
