#include "ObjDistanceFunction.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace {
constexpr uint32_t kLeafSize = 4;

// Parses an OBJ face index ("12", "-1", or "12/3/4"). Returns false on
// malformed input or a zero index. Relative (negative) indices count from the
// vertices read so far, as specified by the OBJ format.
bool parseFaceIndex(const std::string &token, size_t vertexCount, uint32_t &index) {
    const size_t slash = token.find('/');
    const char *begin = token.data();
    const char *end = begin + (slash == std::string::npos ? token.size() : slash);
    if (begin == end) return false;

    long long value = 0;
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end || value == 0) return false;

    const long long resolved = value > 0
        ? value - 1
        : static_cast<long long>(vertexCount) + value;
    if (resolved < 0 || resolved >= static_cast<long long>(vertexCount)) return false;
    index = static_cast<uint32_t>(resolved);
    return true;
}
}  // namespace

bool ObjDistanceFunction::loadObj(const std::string &path, ObjMesh &mesh,
                                  std::string *error) {
    auto fail = [&](const std::string &message) {
        if (error) *error = path + ": " + message;
        return false;
    };

    std::ifstream file(path);
    if (!file) return fail("cannot open file");

    ObjMesh loaded;
    std::string line;
    size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        // Ignore comments and stray CR from Windows files.
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::istringstream stream(line);
        std::string token;
        if (!(stream >> token)) continue;

        if (token == "v") {
            glm::vec3 vertex(0.0f);
            if (!(stream >> vertex.x >> vertex.y >> vertex.z)) {
                return fail("malformed vertex on line " + std::to_string(lineNumber));
            }
            loaded.vertices.push_back(vertex);
        } else if (token == "f") {
            std::vector<uint32_t> face;
            std::string reference;
            while (stream >> reference) {
                uint32_t index = 0;
                if (!parseFaceIndex(reference, loaded.vertices.size(), index)) {
                    return fail("malformed face index '" + reference + "' on line " +
                                std::to_string(lineNumber));
                }
                face.push_back(index);
            }
            if (face.size() < 3) continue;
            // Fan triangulation. Winding is preserved; polygons must be convex,
            // which is what OBJ exporters emit in practice.
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                loaded.triangles.push_back({face[0], face[i], face[i + 1]});
            }
        }
        // vt/vn/o/g/s/usemtl are irrelevant: texture coordinates are ignored
        // and the brush samples the surface with triplanar mapping.
    }

    if (loaded.vertices.empty()) return fail("no vertices");
    if (loaded.triangles.empty()) return fail("no faces");
    mesh = std::move(loaded);
    return true;
}

ObjDistanceFunction::ObjDistanceFunction(const std::string &objPath,
                                         const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    ObjMesh mesh;
    std::string error;
    if (!loadObj(objPath, mesh, &error)) {
        std::cerr << "ObjDistanceFunction: " << error << std::endl;
        return;
    }
    build(mesh, model, bias);
}

void ObjDistanceFunction::build(const ObjMesh &mesh, const Transformation &model,
                                float bias) {
    worldRadius_ = 0.0f;
    if (mesh.vertices.empty() || mesh.triangles.empty()) return;

    // Centre the source mesh on its bounding-box centre and normalize its
    // furthest vertex to the unit sphere. The primitive then fills the same
    // unit-sized cell as the regular polyhedra regardless of the OBJ units.
    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
    for (const glm::vec3 &vertex : mesh.vertices) {
        boundsMin = glm::min(boundsMin, vertex);
        boundsMax = glm::max(boundsMax, vertex);
    }
    const glm::vec3 localCenter = (boundsMin + boundsMax) * 0.5f;

    float localRadius = 0.0f;
    for (const glm::vec3 &vertex : mesh.vertices) {
        localRadius = std::max(localRadius, glm::length(vertex - localCenter));
    }
    if (localRadius <= 1e-8f) return;

    vertices_.resize(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const glm::vec3 local = (mesh.vertices[i] - localCenter) / localRadius;
        vertices_[i] = model.translate + model.quaternion * (local * model.scale);
        worldRadius_ = std::max(worldRadius_, glm::length(vertices_[i] - model.translate));
    }

    triangles_.reserve(mesh.triangles.size());
    for (const Triangle &triangle : mesh.triangles) {
        if (triangle[0] >= vertices_.size() || triangle[1] >= vertices_.size() ||
            triangle[2] >= vertices_.size()) {
            continue;
        }
        // Degenerate faces carry no surface and no usable normal; dropping them
        // also keeps the pseudo-normal sign consistent.
        const glm::vec3 areaNormal = glm::cross(vertices_[triangle[1]] - vertices_[triangle[0]],
                                                vertices_[triangle[2]] - vertices_[triangle[0]]);
        if (glm::dot(areaNormal, areaNormal) <= 1e-24f) continue;
        triangles_.push_back(triangle);
    }
    if (triangles_.empty()) return;

    // Face normals plus angle-weighted vertex pseudo-normals. The angle weight
    // makes the vertex normal independent of how the incident faces happen to
    // be tessellated (Bærentzen & Aanæs).
    faceNormals_.assign(triangles_.size(), glm::vec3(0.0f));
    vertexNormals_.assign(vertices_.size(), glm::vec3(0.0f));
    for (size_t t = 0; t < triangles_.size(); ++t) {
        const Triangle &triangle = triangles_[t];
        const glm::vec3 &a = vertices_[triangle[0]];
        const glm::vec3 &b = vertices_[triangle[1]];
        const glm::vec3 &c = vertices_[triangle[2]];
        glm::vec3 normal = glm::cross(b - a, c - a);
        const float length = glm::length(normal);
        if (length <= 1e-20f) continue;  // degenerate: no normal contribution
        normal /= length;
        faceNormals_[t] = normal;

        const std::array<glm::vec3, 3> edge0 = {
            b - a, c - b, a - c
        };
        const std::array<glm::vec3, 3> edge1 = {
            c - a, a - b, b - c
        };
        for (int corner = 0; corner < 3; ++corner) {
            const glm::vec3 e0 = glm::normalize(edge0[corner]);
            const glm::vec3 e1 = glm::normalize(edge1[corner]);
            const float angle = std::acos(glm::clamp(glm::dot(e0, e1), -1.0f, 1.0f));
            vertexNormals_[triangle[corner]] += normal * angle;
        }
    }
    for (glm::vec3 &normal : vertexNormals_) {
        const float length = glm::length(normal);
        if (length > 1e-20f) normal /= length;
    }

    // Edge pseudo-normal = sum of the incident face normals. Boundary edges
    // fall back to their single face normal.
    edgeNormals_.clear();
    edgeNormalIndex_.clear();
    for (size_t t = 0; t < triangles_.size(); ++t) {
        const Triangle &triangle = triangles_[t];
        for (int edge = 0; edge < 3; ++edge) {
            const uint64_t key = edgeKey(triangle[edge], triangle[(edge + 1) % 3]);
            auto it = edgeNormalIndex_.find(key);
            if (it == edgeNormalIndex_.end()) {
                it = edgeNormalIndex_.emplace(key, static_cast<uint32_t>(edgeNormals_.size())).first;
                edgeNormals_.push_back(glm::vec3(0.0f));
            }
            edgeNormals_[it->second] += faceNormals_[t];
        }
    }
    for (glm::vec3 &normal : edgeNormals_) {
        const float length = glm::length(normal);
        if (length > 1e-20f) normal /= length;
    }

    // Static BVH: median split of the triangle centroids along the longest
    // axis of the node bounds.
    order_.resize(triangles_.size());
    for (uint32_t i = 0; i < order_.size(); ++i) order_[i] = i;
    centroids_.resize(triangles_.size());
    for (size_t t = 0; t < triangles_.size(); ++t) {
        const Triangle &triangle = triangles_[t];
        centroids_[t] = (vertices_[triangle[0]] + vertices_[triangle[1]] +
                         vertices_[triangle[2]]) / 3.0f;
    }

    nodes_.clear();
    nodes_.reserve(triangles_.size() * 2);
    buildNode(0, static_cast<uint32_t>(order_.size()));

    m_sphere = getSphere(model, bias);
}

uint32_t ObjDistanceFunction::buildNode(uint32_t first, uint32_t count) {
    const uint32_t nodeIndex = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();

    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
    for (uint32_t i = first; i < first + count; ++i) {
        const Triangle &triangle = triangles_[order_[i]];
        for (uint32_t index : triangle) {
            boundsMin = glm::min(boundsMin, vertices_[index]);
            boundsMax = glm::max(boundsMax, vertices_[index]);
        }
    }
    nodes_[nodeIndex].min = boundsMin;
    nodes_[nodeIndex].max = boundsMax;

    if (count <= kLeafSize) {
        nodes_[nodeIndex].first = first;
        nodes_[nodeIndex].count = count;
        return nodeIndex;
    }

    glm::vec3 centroidMin(std::numeric_limits<float>::max());
    glm::vec3 centroidMax(std::numeric_limits<float>::lowest());
    for (uint32_t i = first; i < first + count; ++i) {
        centroidMin = glm::min(centroidMin, centroids_[order_[i]]);
        centroidMax = glm::max(centroidMax, centroids_[order_[i]]);
    }
    const glm::vec3 extent = centroidMax - centroidMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    if (extent[axis] <= 1e-12f) {
        // All centroids coincide; splitting would recurse forever.
        nodes_[nodeIndex].first = first;
        nodes_[nodeIndex].count = count;
        return nodeIndex;
    }

    const uint32_t mid = first + count / 2;
    std::nth_element(order_.begin() + first, order_.begin() + mid,
                     order_.begin() + first + count,
                     [&](uint32_t a, uint32_t b) {
                         return centroids_[a][axis] < centroids_[b][axis];
                     });

    const uint32_t left = buildNode(first, mid - first);
    const uint32_t right = buildNode(mid, first + count - mid);
    // Re-index: the recursive calls may have reallocated nodes_.
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    nodes_[nodeIndex].count = 0;
    return nodeIndex;
}

float ObjDistanceFunction::aabbDistanceSq(const BvhNode &node,
                                          const glm::vec3 &p) const {
    const glm::vec3 delta = glm::max(glm::max(node.min - p, p - node.max), glm::vec3(0.0f));
    return glm::dot(delta, delta);
}

uint64_t ObjDistanceFunction::edgeKey(uint32_t a, uint32_t b) {
    const uint32_t lo = std::min(a, b);
    const uint32_t hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | hi;
}

void ObjDistanceFunction::closestPointOnTriangle(const glm::vec3 &p,
                                                 const glm::vec3 &a,
                                                 const glm::vec3 &b,
                                                 const glm::vec3 &c,
                                                 glm::vec3 &closest,
                                                 TriangleRegion &region) {
    const glm::vec3 ab = b - a;
    const glm::vec3 ac = c - a;
    const glm::vec3 areaNormal = glm::cross(ab, ac);
    if (glm::dot(areaNormal, areaNormal) <= 1e-24f) {
        // Degenerate (collinear) triangle: the Voronoi-region math below would
        // divide by a zero barycentric sum. Project onto the three segments.
        const std::array<glm::vec3, 3> points = {a, b, c};
        const std::array<TriangleRegion, 3> edgeRegions = {
            TriangleRegion::EdgeAB, TriangleRegion::EdgeBC, TriangleRegion::EdgeCA
        };
        const std::array<TriangleRegion, 3> startRegions = {
            TriangleRegion::VertexA, TriangleRegion::VertexB, TriangleRegion::VertexC
        };
        const std::array<TriangleRegion, 3> endRegions = {
            TriangleRegion::VertexB, TriangleRegion::VertexC, TriangleRegion::VertexA
        };

        float bestDistanceSq = std::numeric_limits<float>::max();
        for (int edge = 0; edge < 3; ++edge) {
            const glm::vec3 &start = points[edge];
            const glm::vec3 &end = points[(edge + 1) % 3];
            const glm::vec3 segment = end - start;
            const float lengthSq = glm::dot(segment, segment);
            const float t = lengthSq > 1e-24f
                ? glm::clamp(glm::dot(p - start, segment) / lengthSq, 0.0f, 1.0f)
                : 0.0f;
            const glm::vec3 candidate = start + segment * t;
            const glm::vec3 delta = p - candidate;
            const float distanceSq = glm::dot(delta, delta);
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                closest = candidate;
                region = t <= 0.0f ? startRegions[edge]
                       : t >= 1.0f ? endRegions[edge]
                                   : edgeRegions[edge];
            }
        }
        return;
    }

    // Ericson, Real-Time Collision Detection: Voronoi-region classification so
    // the returned point also tells us which pseudo-normal to use for the sign.
    const glm::vec3 ap = p - a;

    const float d1 = glm::dot(ab, ap);
    const float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        closest = a;
        region = TriangleRegion::VertexA;
        return;
    }

    const glm::vec3 bp = p - b;
    const float d3 = glm::dot(ab, bp);
    const float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        closest = b;
        region = TriangleRegion::VertexB;
        return;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        closest = a + ab * v;
        region = TriangleRegion::EdgeAB;
        return;
    }

    const glm::vec3 cp = p - c;
    const float d5 = glm::dot(ab, cp);
    const float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        closest = c;
        region = TriangleRegion::VertexC;
        return;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        closest = a + ac * w;
        region = TriangleRegion::EdgeCA;
        return;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + (c - b) * w;
        region = TriangleRegion::EdgeBC;
        return;
    }

    const float denominator = 1.0f / (va + vb + vc);
    const float v = vb * denominator;
    const float w = vc * denominator;
    closest = a + ab * v + ac * w;
    region = TriangleRegion::Face;
}

const glm::vec3 &ObjDistanceFunction::pseudoNormal(const Triangle &triangle,
                                                   uint32_t triangleIndex,
                                                   TriangleRegion region) const {
    switch (region) {
        case TriangleRegion::VertexA: return vertexNormals_[triangle[0]];
        case TriangleRegion::VertexB: return vertexNormals_[triangle[1]];
        case TriangleRegion::VertexC: return vertexNormals_[triangle[2]];
        case TriangleRegion::EdgeAB: {
            auto it = edgeNormalIndex_.find(edgeKey(triangle[0], triangle[1]));
            if (it != edgeNormalIndex_.end()) return edgeNormals_[it->second];
            break;
        }
        case TriangleRegion::EdgeBC: {
            auto it = edgeNormalIndex_.find(edgeKey(triangle[1], triangle[2]));
            if (it != edgeNormalIndex_.end()) return edgeNormals_[it->second];
            break;
        }
        case TriangleRegion::EdgeCA: {
            auto it = edgeNormalIndex_.find(edgeKey(triangle[2], triangle[0]));
            if (it != edgeNormalIndex_.end()) return edgeNormals_[it->second];
            break;
        }
        case TriangleRegion::Face:
        default:
            return faceNormals_[triangleIndex];
    }
    return faceNormals_[triangleIndex];
}

float ObjDistanceFunction::distance(const glm::vec3 &p) const {
    if (triangles_.empty() || nodes_.empty()) {
        return std::numeric_limits<float>::max();
    }

    float bestDistanceSq = std::numeric_limits<float>::max();
    glm::vec3 bestClosest(0.0f);
    TriangleRegion bestRegion = TriangleRegion::Face;
    uint32_t bestTriangle = 0;

    // Iterative depth-first traversal with a fixed stack. Pushing the farther
    // child first makes the nearer subtree pop first, which shrinks the prune
    // radius quickly.
    uint32_t stack[64];
    uint32_t stackSize = 0;
    stack[stackSize++] = 0;
    while (stackSize > 0) {
        const BvhNode &node = nodes_[stack[--stackSize]];
        if (aabbDistanceSq(node, p) >= bestDistanceSq) continue;

        if (node.isLeaf()) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t triangleIndex = order_[i];
                const Triangle &triangle = triangles_[triangleIndex];
                glm::vec3 closest(0.0f);
                TriangleRegion region = TriangleRegion::Face;
                closestPointOnTriangle(p, vertices_[triangle[0]], vertices_[triangle[1]],
                                       vertices_[triangle[2]], closest, region);
                const glm::vec3 delta = p - closest;
                const float distanceSq = glm::dot(delta, delta);
                if (distanceSq < bestDistanceSq) {
                    bestDistanceSq = distanceSq;
                    bestClosest = closest;
                    bestRegion = region;
                    bestTriangle = triangleIndex;
                }
            }
            continue;
        }

        const float leftDistance = aabbDistanceSq(nodes_[node.left], p);
        const float rightDistance = aabbDistanceSq(nodes_[node.right], p);
        const uint32_t nearChild = leftDistance <= rightDistance ? node.left : node.right;
        const uint32_t farChild = leftDistance <= rightDistance ? node.right : node.left;
        const float nearDistance = std::min(leftDistance, rightDistance);
        const float farDistance = std::max(leftDistance, rightDistance);
        if (farDistance < bestDistanceSq) stack[stackSize++] = farChild;
        if (nearDistance < bestDistanceSq) stack[stackSize++] = nearChild;
    }

    if (bestDistanceSq == std::numeric_limits<float>::max()) {
        return std::numeric_limits<float>::max();
    }

    const float bestDistance = std::sqrt(bestDistanceSq);
    const Triangle &triangle = triangles_[bestTriangle];
    const glm::vec3 &normal = pseudoNormal(triangle, bestTriangle, bestRegion);

    // Pseudo-normal sign test: negative dot means p lies behind the surface.
    return glm::dot(p - bestClosest, normal) < 0.0f ? -bestDistance : bestDistance;
}

BoundingSphere ObjDistanceFunction::getSphere(const Transformation &model,
                                              float bias) const {
    // The (unbiased) surface is contained in the convex hull of the baked
    // vertices, so worldRadius_ is an exact conservative radius under any
    // affine model transform. bias only pads the octree culling bound.
    return BoundingSphere(getCenter(), worldRadius_ + std::max(bias, 0.0f));
}
