#include "EdgeSpan.hpp"
#include "../sdf/SDF.hpp"
#include <cmath>
#include <limits>

namespace {
void edgeAxes(int axis, int &u, int &v) {
    if (axis == 0) {
        u = 1; v = 2;
    } else if (axis == 1) {
        u = 0; v = 2;
    } else {
        u = 0; v = 1;
    }
}

void quadrantSigns(int axis, int quadrant, int &su, int &sv) {
    static const int SIGNS[3][4][2] = {
        {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}},
        {{-1, -1}, { 1, -1}, { 1,  1}, {-1,  1}},
        {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}}
    };
    su = SIGNS[axis][quadrant][0];
    sv = SIGNS[axis][quadrant][1];
}

// One ulp toward a side. A sample sitting exactly on a face shared with the
// neighboring cell is ambiguous (the descent may land in either cell), so it is
// moved into the intended quadrant.
float sideCoordinate(float value, int side, float offset) {
    const float nudged = std::nextafter(value, side < 0 ? -std::numeric_limits<float>::infinity()
                                                        :  std::numeric_limits<float>::infinity());
    // At 0.0 the next float is a denormal. GPUs with FTZ (and some CPUs with
    // DAZ) flush it to zero, so -0.0 compares >= 0.0 and the "negative"
    // quadrant resolves into the positive cell: exactly the missing segments
    // along the world x/y/z axis planes. Use a small normal offset instead
    // whenever the ulp nudge is subnormal.
    if (std::fabs(nudged) < std::numeric_limits<float>::min()) {
        return value + (side < 0 ? -offset : offset);
    }
    return nudged;
}
} // namespace

EdgeSpan EdgeSpan::fromCellEdge(int edgeIndex, const BoundingCube &cellCube, float rootLength) {
    glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
    glm::vec3 p0 = cellCube.getCorner(edgeCorners.x);
    glm::vec3 p1 = cellCube.getCorner(edgeCorners.y);
    glm::vec3 d = glm::abs(p1 - p0);

    EdgeSpan edge;
    if(d.x >= d.y && d.x >= d.z) {
        edge.axis = 0;
    } else if(d.y >= d.x && d.y >= d.z) {
        edge.axis = 1;
    } else {
        edge.axis = 2;
    }
    edgeAxes(edge.axis, edge.u, edge.v);
    edge.fixedU = p0[edge.u];
    edge.fixedV = p0[edge.v];
    edge.start = glm::min(p0[edge.axis], p1[edge.axis]);
    edge.end = glm::max(p0[edge.axis], p1[edge.axis]);
    edge.eps = glm::max(rootLength, cellCube.getLengthX()) * 1e-6f;
    // 0.1% of the cell: enough to survive FTZ at coordinate 0, far below the
    // quadrant cell's size (neighbour cells at the walk resolution are at
    // least this large), so the sample still lands in the intended cell.
    edge.sampleOffset = std::max(cellCube.getLengthX() * 1e-3f, 1e-6f);
    return edge;
}

glm::vec3 EdgeSpan::point(float t) const {
    glm::vec3 p(0.0f);
    p[axis] = t;
    p[u] = fixedU;
    p[v] = fixedV;
    return p;
}

glm::vec3 EdgeSpan::samplePoint(float t, int quadrant) const {
    int su, sv;
    quadrantSigns(axis, quadrant, su, sv);

    glm::vec3 p(0.0f);
    p[axis] = t;
    p[u] = sideCoordinate(fixedU, su, sampleOffset);
    p[v] = sideCoordinate(fixedV, sv, sampleOffset);
    return p;
}
