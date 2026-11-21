#ifndef OCTREE_RAYMARCH_PAYLOAD_GLSL
#define OCTREE_RAYMARCH_PAYLOAD_GLSL

// Device-only derived payload, one entry per stored node. OctreeDerive.comp
// writes it once per table revision; the fragment march reads it only at hits.
// Must match OctreeRaymarchPayload in utils/OctreeRaymarchTable.hpp.
//
// 32 bytes: one 32 B sector per node, so a ring-cell load cannot straddle two
// sectors. The w components are tags for renderdoc/debug inspection: qef.w = 1
// means the entry was written, normal.w is reserved and stays 0.
struct RaymarchPayload {
    vec4 qef;    // xyz = QEF position, w = 1
    vec4 normal; // xyz = QEF normal,   w = 0
};

#endif // OCTREE_RAYMARCH_PAYLOAD_GLSL
