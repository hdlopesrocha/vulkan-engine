#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// One flattened octree node consumed by the SDF raymarch pass
// (shaders/renderer/octree/OctreeRaymarch.frag). Field-for-field mirror of the
// shader's `RaymarchNode` struct (shaders/includes/octree/RaymarchNodeRecord.glsl):
// 8 corner samples, the node cube, plus the material/mapping payload the
// surface shader needs to shade a hit without any rasterized vertex data.
// The table stores SURFACE nodes only (plus one root entry per layer). A
// non-surface child is not stored; its SpaceType is packed into the parent's
// flags so the shader can skip both Empty and Solid leaf regions (neither can
// contain an isosurface crossing). Surface children dropped by the node cap
// are marked in the parent's pruned mask and fall back to the parent's
// trilinear field, which keeps truncation uniform across the tree instead of
// failing the whole table.
//
// Child header (compact): a node's stored Surface children are appended
// contiguously by the builder, so the record only needs the first index
// (meta.y = childBase) plus an 8-bit presence mask (hsv1.w). Slot s resolves
// to childBase + popcount(mask & ((1 << s) - 1)). A node with no stored child
// keeps childBase = 0 and mask = 0.
struct OctreeRaymarchNode {
    glm::vec4 sdf0;        // sdf[0..3]
    glm::vec4 sdf1;        // sdf[4..7]
    glm::vec4 cubeMinSize; // xyz = node cube min, w = edge length
    glm::ivec4 meta;       // x = lod (host) / resolved material (derived), y = childBase,
                           // z = brushIndex, w = flags
    glm::vec4 hsv0;        // hsv.xyz, textureType
    glm::vec4 hsv1;        // data.xy (mapping/UV scale), z = 1/Lipschitz bound (derived),
                           // w = childMask (bit s = slot s has a stored child)
};

static_assert(sizeof(OctreeRaymarchNode) == 96,
              "OctreeRaymarchNode must stay 96 bytes (6 x vec4) to match the "
              "std430 RaymarchNode in RaymarchNodeRecord.glsl");

// One rendered layer root: the node index the shader starts from plus the
// layer's stable id, renderer kind and conservative surface AABB. The id feeds
// the debug view's per-layer hue; `kind` selects the shading branch at a hit
// (0 = solid surface shading, 1 = water surface shading) so one march covers
// every terrain layer. The AABB bounds every surface point the root march can
// sample (union of the stored surface cells, not the whole layer cube), so the
// shader can reject a root or clamp the root's ray interval without ever
// eliding a crossing. `surfaceMin.w`/`surfaceMax.w` are unused padding.
struct OctreeRaymarchRoot {
    uint32_t nodeIndex = 0;
    int32_t layerId = -1;
    uint32_t kind = 0; // 0 = Solid, 1 = Water (LayerRendererType)
    uint32_t pad = 0;
    glm::vec4 surfaceMin = glm::vec4(0.0f); // xyz = surface AABB min
    glm::vec4 surfaceMax = glm::vec4(0.0f); // xyz = surface AABB max
};

static_assert(sizeof(OctreeRaymarchRoot) == 48,
              "OctreeRaymarchRoot must stay 48 bytes (4 x 4 B + 2 x vec4) to "
              "match the std430 RaymarchRoot in RaymarchTable.glsl");

// Device-only derived payload, one per stored node, written by
// OctreeDerive.comp and read by the hit ring. The host never fills this struct;
// it is declared here to keep the GPU stride and field order unambiguous.
// Must match RaymarchPayload in shaders/includes/octree/RaymarchPayload.glsl.
struct OctreeRaymarchPayload {
    glm::vec4 qef;    // xyz = QEF position, w = 1
    glm::vec4 normal; // xyz = QEF normal, w = 0
};

static_assert(sizeof(OctreeRaymarchPayload) == 32,
              "OctreeRaymarchPayload must stay 32 bytes (2 x vec4) to match "
              "the std430 RaymarchPayload in RaymarchPayload.glsl");

// Phase 0 instrumentation counters, one block per frame slot. Only the
// OCTREE_PROFILE fragment variant touches this buffer (fragment atomics
// gated by the push-constant flag), so the production variants pay no
// atomics. Layout must match shaders/includes/octree/RaymarchCounters.glsl.
struct OctreeRaymarchCounters {
    uint32_t rays = 0;                 // fragment invocations
    uint32_t rootTests = 0;            // root AABB / interval tests
    uint32_t rootMarches = 0;          // rmMarchRoot calls that entered
    uint32_t nodeVisits = 0;           // distinct node entries (push/pop)
    uint32_t leafIters = 0;            // iterations with leaf == true
    uint32_t nonLeafIters = 0;         // iterations that descend/test a child
    uint32_t fieldEvals = 0;           // march rmTrilinear calls
    uint32_t gradientEvals = 0;        // rmNodeNormal calls
    uint32_t qefSolves = 0;            // sdfCellPosition calls (fallback)
    uint32_t qefPayloadLoads = 0;      // rmPayload[id].qef reads
    uint32_t ringDescents = 0;         // root-fallback descents from the hit path
    uint32_t ringCells = 0;            // ring cells considered
    uint32_t ringDuplicates = 0;       // ring cells dropped as duplicates
    uint32_t materialClassifiers = 0;  // rmResolveMaterialAt fallback calls
    uint32_t carryHits = 0;            // carry-forward reuses
    uint32_t earlyRoots = 0;           // roots skipped by AABB/order
    uint32_t bisectEvals = 0;          // field evaluations inside bisection
    uint32_t waterSecondary = 0;       // secondary SDF root-list marches
    uint32_t hits = 0;                 // accepted primary hits
    uint32_t discardSkips = 0;         // hits rejected by material / back face
    uint32_t enabled = 0;              // CPU-written gate: 1 = counters on
};
static_assert(sizeof(OctreeRaymarchCounters) == 84,
              "OctreeRaymarchCounters must stay 21 x uint32 to match "
              "shaders/includes/octree/RaymarchCounters.glsl");

// CPU-side owner of one raymarch table revision. The owner (MyApp) rebuilds it
// on a debounce and hands it to OctreeRaymarchRenderer, which uploads the same
// revision into each frame slot exactly once (version match = already there).
class OctreeRaymarchTable {
public:
    // meta.w packing, shared with shaders/includes/octree/RaymarchNodeRecord.glsl:
    //   bit0       leaf (no children block)
    //   bit1       derived payload valid (RM_FLAG_PAYLOAD): meta.x = resolved
    //              material index, hsv1.z = 1/Lipschitz bound. Written by the
    //              renderer's once-per-revision OctreeDerive.comp pass.
    //   bit2       QEF payload valid (RM_FLAG_QEF): OctreeRaymarchPayload[node]
    //              holds the QEF position/normal. Also written by the derive
    //              pass, in the same invocation as bit1.
    //   bits8..15  mask of surface children pruned by the node cap (per slot)
    //   bits16..31 two-bit SpaceType per child slot (non-surface leaves only)
    // meta.y is the stored-child base (NOT a SpaceType): all stored nodes are
    // Surface by construction, and their children are contiguous.
    static constexpr uint32_t FLAG_LEAF = 1u;
    static constexpr uint32_t FLAG_PAYLOAD = 1u << 1;
    static constexpr uint32_t FLAG_QEF = 1u << 2;
    static constexpr uint32_t PRUNED_CHILD_SHIFT = 8u;
    static constexpr uint32_t CHILD_TYPE_SHIFT = 16u;

    void clear();

    // Append one record and return its table index. Callers patch parent child
    // links by index afterwards: the vector may reallocate between appends.
    uint32_t append(const OctreeRaymarchNode& node);

    void setChild(uint32_t parent, int slot, int32_t child);
    void setPrunedChild(uint32_t parent, int slot);
    void setChildType(uint32_t parent, int slot, uint32_t type);

    std::vector<OctreeRaymarchNode> nodes;
    // Root entry per rendered layer (node index + stable Layer::Id + surface
    // AABB). The shader sorts the roots by AABB entry, marches them front to
    // back and clamps every later march to the nearest hit so far; the id
    // feeds the per-layer debug tint.
    std::vector<OctreeRaymarchRoot> layerRoots;
    // Monotonic revision. Bumped by the owner on every rebuild (including an
    // empty result) so stale per-slot uploads are cleared.
    uint64_t version = 0;
};
