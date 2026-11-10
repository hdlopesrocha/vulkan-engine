#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// One flattened octree node consumed by the SDF raymarch pass
// (shaders/renderer/octree/OctreeRaymarch.frag). Field-for-field mirror of the
// shader's `RaymarchNode` struct (shaders/includes/octree/RaymarchTable.glsl):
// 8 corner samples, the node cube, plus the material/mapping payload the
// surface shader needs to shade a hit without any rasterized vertex data.
// The table stores SURFACE nodes only (plus one root entry per layer). A
// non-surface child is not stored; its SpaceType is packed into the parent's
// flags so the shader can skip both Empty and Solid leaf regions (neither can
// contain an isosurface crossing). Surface children dropped by the node cap
// are marked in the parent's pruned mask and fall back to the parent's
// trilinear field, which keeps truncation uniform across the tree instead of
// failing the whole table.
struct OctreeRaymarchNode {
    glm::vec4 sdf0;        // sdf[0..3]
    glm::vec4 sdf1;        // sdf[4..7]
    glm::vec4 cubeMinSize; // xyz = node cube min, w = edge length
    glm::ivec4 meta;       // x = lod, y = SpaceType, z = brushIndex, w = flags
    glm::ivec4 child03;    // child index per corner slot 0..3 (-1 = absent)
    glm::ivec4 child47;    // child index per corner slot 4..7 (-1 = absent)
    glm::vec4 hsv0;        // hsv.xyz, textureType
    glm::vec4 hsv1;        // data.xy (mapping/UV scale), 0, 0
};

static_assert(sizeof(OctreeRaymarchNode) == 128,
              "OctreeRaymarchNode must stay 128 bytes (8 x vec4) to match the "
              "std430 RaymarchNode in RaymarchTable.glsl");

// One rendered layer root: the node index the shader starts from plus the
// layer's stable id. The id only feeds the debug view's per-layer hue (and
// any future per-layer diagnostics); the march itself ignores it.
struct OctreeRaymarchRoot {
    uint32_t nodeIndex = 0;
    int32_t layerId = -1;
};

static_assert(sizeof(OctreeRaymarchRoot) == 8,
              "OctreeRaymarchRoot must match the std430 RaymarchRoot in "
              "RaymarchTable.glsl");

// CPU-side owner of one raymarch table revision. The owner (MyApp) rebuilds it
// on a debounce and hands it to OctreeRaymarchRenderer, which uploads the same
// revision into each frame slot exactly once (version match = already there).
class OctreeRaymarchTable {
public:
    // meta.w packing, shared with shaders/includes/octree/RaymarchTable.glsl:
    //   bit0       leaf (no children block)
    //   bits8..15  mask of surface children pruned by the node cap (per slot)
    //   bits16..31 two-bit SpaceType per child slot (non-surface leaves only)
    static constexpr uint32_t FLAG_LEAF = 1u;
    static constexpr uint32_t PRUNED_CHILD_SHIFT = 8u;
    static constexpr uint32_t CHILD_TYPE_SHIFT = 16u;
    static constexpr int32_t NO_CHILD = -1;

    void clear();

    // Append one record and return its table index. Callers patch parent child
    // links by index afterwards: the vector may reallocate between appends.
    uint32_t append(const OctreeRaymarchNode& node);

    void setChild(uint32_t parent, int slot, int32_t child);
    void setPrunedChild(uint32_t parent, int slot);
    void setChildType(uint32_t parent, int slot, uint32_t type);

    std::vector<OctreeRaymarchNode> nodes;
    // Root entry per rendered layer (node index + stable Layer::Id). The
    // shader ray-boxes each root cube and keeps the nearest hit across all of
    // them; the id feeds the per-layer debug tint.
    std::vector<OctreeRaymarchRoot> layerRoots;
    // Monotonic revision. Bumped by the owner on every rebuild (including an
    // empty result) so stale per-slot uploads are cleared.
    uint64_t version = 0;
};
