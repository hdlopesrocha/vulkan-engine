#pragma once
#include <glm/glm.hpp>
#include "../math/geometry/BoundingCube.hpp"

// A sub-span of one cube edge: the edge's axis, the two other axes (u, v),
// where the edge sits on those axes (fixedU/fixedV) and how far it extends
// along the axis (start..end). eps is the degeneracy tolerance used when
// splitting at neighboring LOD boundaries.
//
// Extracted from OctreeTriangleTessellator: the span carries its own sampling
// helpers, so the walk only has to decide *where* along it to sample.
struct EdgeSpan {
    int axis = 0;
    int u = 1;
    int v = 2;
    float fixedU = 0.0f;
    float fixedV = 0.0f;
    float start = 0.0f;
    float end = 0.0f;
    float eps = 1e-6f;
    // Side-selector nudge for samplePoint(). One ulp is a denormal at exactly
    // 0.0, and GPUs with denormals-flush-to-zero collapse +/- there, merging
    // the quadrants. This is a small NORMAL offset (< 1% of the cell) so the
    // sample stays inside the intended quadrant cell on every backend.
    float sampleOffset = 1e-6f;

    // Span of the SDF corner-pair edge `edgeIndex` on `cellCube`; the axis is
    // the edge's dominant direction. `rootLength` is the world root's edge
    // length: eps scales with the larger of root and cell so the tolerance
    // stays meaningful at every LoD level.
    static EdgeSpan fromCellEdge(int edgeIndex, const BoundingCube &cellCube, float rootLength);

    // Point on the span's line at axis parameter t (fixedU/fixedV on the other
    // two axes).
    glm::vec3 point(float t) const;

    // Same, but nudged one ulp off the fixedU/fixedV planes toward `quadrant`
    // (0..3) so the sample lands strictly inside the intended quadrant cell
    // rather than exactly on a face shared with its neighbor.
    glm::vec3 samplePoint(float t, int quadrant) const;
};
