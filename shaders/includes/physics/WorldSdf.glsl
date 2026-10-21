// World-space SDF query over the flattened octree node table.
//
// Generic and self-contained: it declares the node-table storage buffers
// (set 0 bindings 4..9, the same layout shaders/computation/octree/
// OctreeTriangles.comp consumes) and the helpers every consumer needs —
// descent to the finest node containing a point, trilinear corner
// interpolation and the finite-difference gradient. Included by
// includes/physics/ParticlePhysics.glsl; reusable by any compute shader that
// binds a flattened octree region (particle physics, future rigid-body
// queries, brush/probe tools).

#extension GL_EXT_scalar_block_layout : enable

// Octree node table (entry 0 = world root; see OctreeGpuNode).
layout(std430, set = 0, binding = 4, scalar) readonly buffer InSdf0    { vec4 sdf0[]; };       // sdf[0..3]
layout(std430, set = 0, binding = 5, scalar) readonly buffer InSdf1    { vec4 sdf1[]; };       // sdf[4..7]
layout(std430, set = 0, binding = 6, scalar) readonly buffer InCube   { vec4 cubeMinSize[]; }; // xyz min, w size
layout(std430, set = 0, binding = 7, scalar) readonly buffer InMeta   { ivec4 meta[]; };       // (lod, type, brushIndex, flags)
layout(std430, set = 0, binding = 8, scalar) readonly buffer InChild03 { ivec4 child03[]; };
layout(std430, set = 0, binding = 9, scalar) readonly buffer InChild47 { ivec4 child47[]; };


// Octree node types (space/SpaceType.hpp): 0 empty, 1 surface, 2 solid.
const int TYPE_EMPTY = 0;
const int TYPE_SURFACE = 1;
const int TYPE_SOLID = 2;

// Distance returned when the world SDF has no data at a sample (missing child
// or no finite corner): far beyond any droplet radius, so the sample counts as
// "no collision" without ever turning into INFINITY/NaN arithmetic.
const float SDF_NO_DATA = 1e6;


// Defensive bound on the octree descent: a malformed node table must never be
// able to spin the shader forever (valid trees stop at leaves long before).
const int MAX_DESCENT_DEPTH = 32;


// ---------------------------------------------------------------- node data

float nodeSdf(uint id, int idx) {
    vec4 v = (idx < 4) ? sdf0[id] : sdf1[id];
    return v[idx & 3];
}
int nodeType(uint id) { return meta[id].y; }
bool nodeIsLeaf(uint id) { return (meta[id].w & 1) != 0; }
vec3 nodeMin(uint id) { return cubeMinSize[id].xyz; }
float nodeSize(uint id) { return cubeMinSize[id].w; }
int nodeChild(uint id, int i) {
    ivec4 c = (i < 4) ? child03[id] : child47[id];
    return c[i & 3];
}


bool isFiniteF(float v) { return !(isnan(v) || isinf(v)); }
bool allFinite(vec3 v) { return isFiniteF(v.x) && isFiniteF(v.y) && isFiniteF(v.z); }


// -------------------------------------------------------- world SDF lookup

// Octree::iterateTriangles::findCellAt descent, without a target LoD: walk from
// the world root (entry 0) down to the finest node whose cube contains pos and
// stop at leaves. Returns false when a child is absent (the sample is outside
// the uploaded table), when the root does not contain pos, or when the descent
// hits its defensive depth cap without reaching a leaf.
bool sdfNodeAt(vec3 pos, out uint nodeId) {
    nodeId = 0u;

    vec3 rootMin = nodeMin(0u);
    float rootSize = nodeSize(0u);
    if (any(lessThan(pos, rootMin)) || any(greaterThan(pos, rootMin + vec3(rootSize)))) {
        return false; // outside the octree world: no terrain to collide with
    }

    uint idx = 0u;
    for (int depth = 0; depth < MAX_DESCENT_DEPTH; ++depth) {
        if (nodeIsLeaf(idx)) {
            break;
        }
        // Same child selection arithmetic as findCellAt: bit2 = x, bit1 = y,
        // bit0 = z, corner at min + 0.5 * size splits the cube.
        vec3 mn = nodeMin(idx);
        float halfSize = nodeSize(idx) * 0.5;
        int childIndex = (pos.x >= mn.x + halfSize ? 4 : 0)
                       + (pos.y >= mn.y + halfSize ? 2 : 0)
                       + (pos.z >= mn.z + halfSize ? 1 : 0);
        int child = nodeChild(idx, childIndex);
        if (child < 0) {
            return false; // child absent from the table
        }
        idx = uint(child);
    }

    nodeId = idx;
    return true;
}


// World signed distance at `pos` (trilinear over the containing node's corners),
// plus the cube size of that node for the gradient step. INFINITY corners are
// "no data": they are replaced by the average of the finite corners so they
// cannot poison the interpolation; a node with no finite corner at all (or an
// empty node) returns SDF_NO_DATA, i.e. no collision.
float sdfAt(vec3 pos, out float outCellSize) {
    outCellSize = 1.0;

    uint id;
    if (!sdfNodeAt(pos, id)) {
        return SDF_NO_DATA;
    }
    float sz = nodeSize(id);
    if (!(sz > 0.0)) {
        return SDF_NO_DATA;
    }
    outCellSize = sz;
    if (nodeType(id) == TYPE_EMPTY) {
        return SDF_NO_DATA; // no surface inside: nothing to push out of
    }

    float c[8];
    float sum = 0.0;
    int finiteCount = 0;
    for (int i = 0; i < 8; ++i) {
        float v = nodeSdf(id, i);
        if (isFiniteF(v)) {
            c[i] = v;
            sum += v;
            ++finiteCount;
        } else {
            c[i] = 0.0; // placeholder, replaced by the finite average below
        }
    }
    if (finiteCount == 0) {
        return SDF_NO_DATA;
    }
    if (finiteCount < 8) {
        float avg = sum / float(finiteCount);
        for (int i = 0; i < 8; ++i) {
            if (!isFiniteF(nodeSdf(id, i))) {
                c[i] = avg;
            }
        }
    }

    // Trilinear interpolation inside the node cube (corner bit convention:
    // bit2 = x, bit1 = y, bit0 = z — same ordering as SDF::interpolate).
    vec3 local = clamp((pos - nodeMin(id)) / sz, vec3(0.0), vec3(1.0));
    float v000 = mix(c[0], c[1], local.z);
    float v010 = mix(c[2], c[3], local.z);
    float v100 = mix(c[4], c[5], local.z);
    float v110 = mix(c[6], c[7], local.z);
    float v00 = mix(v000, v010, local.y);
    float v10 = mix(v100, v110, local.y);
    return mix(v00, v10, local.x);
}


// Outward surface normal: central differences of the SDF, normalized. The
// step is scaled by the containing node's cube size (clamped to 0.05 m so a
// tiny leaf still gets a usable offset). A degenerate gradient falls back to
// straight up, which is the only direction that is always "out" of the
// horizontal terrain and of the water plane.
vec3 sdfGradient(vec3 pos, float cellSize) {
    float eps = max(0.05, 0.05 * cellSize);
    float unused;
    vec3 g;
    g.x = sdfAt(pos + vec3(eps, 0.0, 0.0), unused) - sdfAt(pos - vec3(eps, 0.0, 0.0), unused);
    g.y = sdfAt(pos + vec3(0.0, eps, 0.0), unused) - sdfAt(pos - vec3(0.0, eps, 0.0), unused);
    g.z = sdfAt(pos + vec3(0.0, 0.0, eps), unused) - sdfAt(pos - vec3(0.0, 0.0, eps), unused);
    float len = length(g);
    if (!isFiniteF(len) || len < 1e-6) {
        return vec3(0.0, 1.0, 0.0);
    }
    return g / len;
}
