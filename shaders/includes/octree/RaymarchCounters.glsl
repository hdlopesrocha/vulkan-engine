#ifndef OCTREE_RAYMARCH_COUNTERS_GLSL
#define OCTREE_RAYMARCH_COUNTERS_GLSL

// Phase 0 instrumentation: counter indices shared by the profile fragment
// variant and the host-side OctreeRaymarchCounters struct
// (utils/OctreeRaymarchTable.hpp). The production fragment variants declare
// the binding but never access it; only OctreeRaymarchProfile*.frag.spv emits
// the atomics, and only while the push-constant gate is set.
#define RM_COUNTER_RAYS              0
#define RM_COUNTER_ROOT_TESTS        1
#define RM_COUNTER_ROOT_MARCHES      2
#define RM_COUNTER_NODE_VISITS       3
#define RM_COUNTER_LEAF_ITERS        4
#define RM_COUNTER_NONLEAF_ITERS     5
#define RM_COUNTER_FIELD_EVALS       6
#define RM_COUNTER_GRADIENT_EVALS    7
#define RM_COUNTER_QEF_SOLVES        8
#define RM_COUNTER_QEF_PAYLOAD       9
#define RM_COUNTER_RING_DESCENTS     10
#define RM_COUNTER_RING_CELLS        11
#define RM_COUNTER_RING_DUPLICATES   12
#define RM_COUNTER_MATERIAL_CLASSIFY 13
#define RM_COUNTER_CARRY_HITS        14
#define RM_COUNTER_EARLY_ROOTS       15
#define RM_COUNTER_BISECT_EVALS      16
#define RM_COUNTER_WATER_SECONDARY   17
#define RM_COUNTER_HITS              18
#define RM_COUNTER_DISCARD_SKIPS     19
#define RM_COUNTER_ENABLED           20
#define RM_COUNTER_COUNT             21

#endif // OCTREE_RAYMARCH_COUNTERS_GLSL
