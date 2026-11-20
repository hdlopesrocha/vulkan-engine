#ifndef OCTREE_RAYMARCH_NODE_RECORD_GLSL
#define OCTREE_RAYMARCH_NODE_RECORD_GLSL

// One flattened octree node (128 bytes), shared by the raymarch fragment
// shader (RaymarchTable.glsl) and the once-per-revision payload derive pass
// (shaders/computation/octree/OctreeDerive.comp). Keeping one declaration
// stops the two stages from drifting apart.
//
// Field order/size must match OctreeRaymarchNode
// (utils/OctreeRaymarchTable.hpp).
//
// std430 is sufficient here: every member is a vec4/ivec4, so the scalar and
// std430 layouts are byte-identical and the shader does not depend on the
// optional scalarBlockLayout device feature.
struct RaymarchNode {
    vec4  sdf0;        // sdf[0..3]
    vec4  sdf1;        // sdf[4..7]
    vec4  cubeMinSize; // xyz = node cube min, w = edge length
    ivec4 meta;        // x = lod / resolved material, y = SpaceType, z = brushIndex, w = flags
    ivec4 child03;     // child index per corner slot 0..3 (-1 = absent)
    ivec4 child47;     // child index per corner slot 4..7 (-1 = absent)
    vec4  hsv0;        // hsv.xyz, textureType
    vec4  hsv1;        // data.xy, 1/Lipschitz bound (derived payload), 0
};

// meta.w packing, shared with OctreeRaymarchTable.hpp:
//   bit0       leaf (no children block)
//   bit1       derived payload valid: meta.x = resolved material index,
//              hsv1.z = 1/Lipschitz bound (written by OctreeDerive.comp)
//   bits8..15  mask of surface children pruned by the node cap (per slot)
//   bits16..31 two-bit SpaceType per child slot (non-surface leaves only)
const uint RM_FLAG_LEAF = 1u;
const uint RM_FLAG_PAYLOAD = 2u;

#endif // OCTREE_RAYMARCH_NODE_RECORD_GLSL
