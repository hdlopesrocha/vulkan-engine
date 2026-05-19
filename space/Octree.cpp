#include "Octree.hpp"
#include <memory>
#include <atomic>
#include "../math/BrushMode.hpp"
#include "NodeOperationResult.hpp"
#include "OctreeNodeCubeSerialized.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>

const float INFINITY_ARRAY [8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
#include "OctreeAllocator.hpp"
#include "OctreeNode.hpp"
#include "IteratorHandler.hpp"
#include "../sdf/SDF.hpp"
#include "../math/BrushMode.hpp"


//      6-----7
//     /|    /|
//    4z+---5 |
//    | 2y--+-3
//    |/    |/
//    0-----1x

static std::vector<glm::ivec4> TESSELATION_ORDERS;
static std::vector<glm::ivec2> TESSELATION_EDGES;
static bool initialized = false;

static void initialize() {
    if(!initialized) {
        TESSELATION_ORDERS.push_back(glm::ivec4(0,1,3,2));
        TESSELATION_EDGES.push_back(SDF_EDGES[11]);
        
        TESSELATION_ORDERS.push_back(glm::ivec4(0,2,6,4));
        TESSELATION_EDGES.push_back(SDF_EDGES[6]);
        
        TESSELATION_ORDERS.push_back(glm::ivec4(0,4,5,1));
        TESSELATION_EDGES.push_back(SDF_EDGES[5]);
        initialized = true;
    }
}
Octree::Octree(BoundingCube minCube, float chunkSize) : BoundingCube(minCube), allocator(new OctreeAllocator()) {
    this->chunkSize = chunkSize;
	this->root = allocator->allocate()->init(glm::vec3(minCube.getCenter()));
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
	initialize();
}

Octree::Octree() : Octree(glm::vec3(0.0f), 1.0f) {
    this->chunkSize = 1.0f;
    this->root = NULL;
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
    initialize();
}

int getNodeIndex(const glm::vec3 &vec, const BoundingCube &cube) {
	glm::vec3 c = cube.getCenter();
    return (vec.x >= c.x ? 4 : 0) + (vec.y >= c.y ? 2 : 0) + (vec.z >= c.z ? 1 : 0);
}




OctreeNodeLevel Octree::getNodeAt(const glm::vec3 &pos, int level, bool simplification) const{
    OctreeNode * candidate = root;
    OctreeNode* node = candidate;
    BoundingCube cube = *this;
    uint currentLevel = 0;
	if(!contains(pos)) {
		return OctreeNodeLevel(NULL, 0);
	}
    while (candidate != NULL && level-- > 0 ) {
        if (simplification && node->isSimplified()) {
            break;
        }
        int i = getNodeIndex(pos, cube);
        cube = cube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
        if(candidate != NULL) {
            node = candidate;
            ++currentLevel;
        }
    }
    return OctreeNodeLevel(node, currentLevel);
}

OctreeNode* Octree::getNodeAt(const glm::vec3 &pos, bool simplification) const {
    OctreeNode * candidate = root;
    OctreeNode* node = candidate;
    BoundingCube cube = *this;
	if(!contains(pos)) {
		return NULL;
	}
    while (candidate != NULL) {
        if (simplification && node->isSimplified()) {
            break;
        }
        int i = getNodeIndex(pos, cube);
        cube = cube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
        if(candidate != NULL) {
            node = candidate;
        }
    }
    return node;
}

float Octree::getSdfAt(const glm::vec3 &pos) {
    OctreeNode * candidate = root;
    OctreeNode * node = candidate;
    BoundingCube candidateCube = *this;
    BoundingCube nodeCube = candidateCube;

	if(!contains(pos)) {
		return INFINITY;
	}
    while (candidate) {
        node = candidate;
        nodeCube = candidateCube;
        int i = getNodeIndex(pos, candidateCube);
        candidateCube = nodeCube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
    }

    if(node) {
        return SDF::interpolate(node->sdf, pos, nodeCube);
    }
    std::cerr << "Not interpolated" << std::endl;
    return INFINITY;
}

template <typename T, std::size_t N> 
bool allDifferent(const T (&arr)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (arr[i] == arr[j]) return false;
        }
    }
    return true;
}

template <typename T, typename... Args>
bool allDifferent(const T& first, const Args&... args) {
    std::array<T, sizeof...(args) + 1> arr { first, args... };

    for (std::size_t i = 0; i < arr.size(); ++i) {
        for (std::size_t j = i + 1; j < arr.size(); ++j) {
            if (arr[i] == arr[j]) return false;
        }
    }
    return true;
}

void Octree::iterateTriangles(
        OctreeNode * from,
            const BoundingCube &fromCube,
            int fromLevel,
            OctreeNodeTriangleHandler &func,
            ThreadContext * context) const {
    (void)fromLevel;
    (void)context;

    struct EdgeCell {
        OctreeNode *node = NULL;
        BoundingCube cube;
        int level = 0;

        bool isSurface() const {
            return node != NULL && node->getType() == SpaceType::Surface && node->isSimplified();
        }
    };

    struct EdgeSpan {
        int axis = 0;
        int u = 1;
        int v = 2;
        float fixedU = 0.0f;
        float fixedV = 0.0f;
        float start = 0.0f;
        float end = 0.0f;
        float eps = 1e-6f;
    };

    auto samePosition = [](const glm::vec3 &a, const glm::vec3 &b, float eps) {
        glm::vec3 d = a - b;
        return glm::dot(d, d) <= eps * eps;
    };

    auto edgeAxes = [](int axis, int &u, int &v) {
        if (axis == 0) {
            u = 1; v = 2;
        } else if (axis == 1) {
            u = 0; v = 2;
        } else {
            u = 0; v = 1;
        }
    };

    auto findCellAt = [this](const glm::vec3 &pos) {
        EdgeCell result;
        if(root == NULL || !contains(pos)) {
            return result;
        }

        OctreeNode *node = root;
        BoundingCube cube = *this;
        int level = 0;

        while(node != NULL && !node->isSimplified() && !node->isLeaf()) {
            ChildBlock *block = node->getBlock(*allocator);
            if(block == NULL) {
                node = NULL;
                break;
            }

            int childIndex = getNodeIndex(pos, cube);
            OctreeNode *child = block->get(childIndex, *allocator);
            if(child == NULL) {
                node = NULL;
                break;
            }

            cube = cube.getChild(childIndex);
            node = child;
            ++level;
        }

        if(node != NULL) {
            result.node = node;
            result.cube = cube;
            result.level = level;
        }
        return result;
    };

    auto sideCoordinate = [](float value, int side) {
        return std::nextafter(value, side < 0 ? -std::numeric_limits<float>::infinity()
                                               :  std::numeric_limits<float>::infinity());
    };

    auto quadrantSigns = [](int axis, int quadrant, int &su, int &sv) {
        static const int SIGNS[3][4][2] = {
            {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}},
            {{-1, -1}, { 1, -1}, { 1,  1}, {-1,  1}},
            {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}}
        };
        su = SIGNS[axis][quadrant][0];
        sv = SIGNS[axis][quadrant][1];
    };

    auto edgePoint = [](const EdgeSpan &edge, float t) {
        glm::vec3 p(0.0f);
        p[edge.axis] = t;
        p[edge.u] = edge.fixedU;
        p[edge.v] = edge.fixedV;
        return p;
    };

    auto edgeSamplePoint = [&](const EdgeSpan &edge, float t, int quadrant) {
        int su, sv;
        quadrantSigns(edge.axis, quadrant, su, sv);

        glm::vec3 p(0.0f);
        p[edge.axis] = t;
        p[edge.u] = sideCoordinate(edge.fixedU, su);
        p[edge.v] = sideCoordinate(edge.fixedV, sv);
        return p;
    };

    auto addBreak = [](std::vector<float> &breaks, float value, float start, float end, float eps) {
        if(value <= start + eps || value >= end - eps) {
            return false;
        }
        for(float existing : breaks) {
            if(std::fabs(existing - value) <= eps) {
                return false;
            }
        }
        breaks.push_back(value);
        return true;
    };

    auto sortUnique = [](std::vector<float> &values, float eps) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(), [eps](float a, float b) {
            return std::fabs(a - b) <= eps;
        }), values.end());
    };

    std::function<void(const EdgeSpan&, float, float, std::vector<float>&, int)> collectBreaks;
    collectBreaks = [&](const EdgeSpan &edge, float start, float end, std::vector<float> &breaks, int depth) {
        if(depth > 64 || end - start <= edge.eps * 2.0f) {
            return;
        }

        float mid = start + (end - start) * 0.5f;
        std::vector<float> localBreaks = {start, end};

        for(int q = 0; q < 4; ++q) {
            EdgeCell cell = findCellAt(edgeSamplePoint(edge, mid, q));
            if(cell.node == NULL) {
                continue;
            }

            addBreak(localBreaks, cell.cube.getMin()[edge.axis], start, end, edge.eps);
            addBreak(localBreaks, cell.cube.getMax()[edge.axis], start, end, edge.eps);
        }

        sortUnique(localBreaks, edge.eps);
        if(localBreaks.size() <= 2) {
            return;
        }

        for(size_t i = 1; i + 1 < localBreaks.size(); ++i) {
            addBreak(breaks, localBreaks[i], start, end, edge.eps);
        }

        for(size_t i = 1; i < localBreaks.size(); ++i) {
            collectBreaks(edge, localBreaks[i - 1], localBreaks[i], breaks, depth + 1);
        }
    };

    auto makeEdgeSpan = [&](int edgeIndex) {
        glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
        glm::vec3 p0 = fromCube.getCorner(edgeCorners.x);
        glm::vec3 p1 = fromCube.getCorner(edgeCorners.y);
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
        edge.eps = glm::max(getLengthX(), fromCube.getLengthX()) * 1e-6f;
        return edge;
    };

    auto ownerLess = [](const EdgeCell &a, const EdgeCell &b) {
        float al = a.cube.getLengthX();
        float bl = b.cube.getLengthX();
        if(al != bl) return al < bl;

        glm::vec3 amin = a.cube.getMin();
        glm::vec3 bmin = b.cube.getMin();
        if(amin.x != bmin.x) return amin.x < bmin.x;
        if(amin.y != bmin.y) return amin.y < bmin.y;
        if(amin.z != bmin.z) return amin.z < bmin.z;
        return std::less<OctreeNode*>()(a.node, b.node);
    };

    auto emitTriangle = [&](Vertex *a, Vertex *b, Vertex *c, float eps) {
        if(a == NULL || b == NULL || c == NULL) return;
        if(samePosition(a->position, b->position, eps)) return;
        if(samePosition(b->position, c->position, eps)) return;
        if(samePosition(c->position, a->position, eps)) return;
        func.handle(*a, *b, *c);
    };

    auto emitSegment = [&](const EdgeSpan &edge, float start, float end) {
        float mid = start + (end - start) * 0.5f;
        EdgeCell cells[4];
        for(int q = 0; q < 4; ++q) {
            cells[q] = findCellAt(edgeSamplePoint(edge, mid, q));
            if(!cells[q].isSurface()) {
                return;
            }
        }

        EdgeCell owner = cells[0];
        for(int q = 1; q < 4; ++q) {
            if(ownerLess(cells[q], owner)) {
                owner = cells[q];
            }
        }

        if(owner.node != from) {
            return;
        }

        glm::vec3 p0 = edgePoint(edge, start);
        glm::vec3 p1 = edgePoint(edge, end);
        float d0 = SDF::interpolate(owner.node->sdf, p0, owner.cube);
        float d1 = SDF::interpolate(owner.node->sdf, p1, owner.cube);
        if((d0 < 0.0f) == (d1 < 0.0f)) {
            return;
        }

        std::vector<EdgeCell> polygon;
        polygon.reserve(4);
        for(int q = 0; q < 4; ++q) {
            bool duplicate = !polygon.empty()
                && (polygon.back().node == cells[q].node
                    || samePosition(polygon.back().node->vertex.position, cells[q].node->vertex.position, edge.eps));
            if(!duplicate) {
                polygon.push_back(cells[q]);
            }
        }

        if(polygon.size() > 1) {
            const EdgeCell &first = polygon.front();
            const EdgeCell &last = polygon.back();
            if(first.node == last.node || samePosition(first.node->vertex.position, last.node->vertex.position, edge.eps)) {
                polygon.pop_back();
            }
        }

        for(size_t i = 0; i < polygon.size(); ++i) {
            for(size_t j = i + 1; j < polygon.size(); ++j) {
                if(polygon[i].node == polygon[j].node
                    || samePosition(polygon[i].node->vertex.position, polygon[j].node->vertex.position, edge.eps)) {
                    return;
                }
            }
        }

        if(polygon.size() == 3) {
            emitTriangle(&polygon[0].node->vertex, &polygon[1].node->vertex, &polygon[2].node->vertex, edge.eps);
        } else if(polygon.size() == 4) {
            emitTriangle(&polygon[0].node->vertex, &polygon[1].node->vertex, &polygon[2].node->vertex, edge.eps);
            emitTriangle(&polygon[0].node->vertex, &polygon[2].node->vertex, &polygon[3].node->vertex, edge.eps);
        }
    };

    if(from == NULL || from->getType() != SpaceType::Surface || !from->isSimplified()) {
        return;
    }

    for(int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
        glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
        bool sign0 = from->sdf[edgeCorners.x] < 0.0f;
        bool sign1 = from->sdf[edgeCorners.y] < 0.0f;
        if(sign0 == sign1) {
            continue;
        }

        EdgeSpan edge = makeEdgeSpan(edgeIndex);
        if(edge.end - edge.start <= edge.eps * 2.0f) {
            continue;
        }

        std::vector<float> breaks = {edge.start, edge.end};
        collectBreaks(edge, edge.start, edge.end, breaks, 0);
        sortUnique(breaks, edge.eps);

        for(size_t i = 1; i < breaks.size(); ++i) {
            if(breaks[i] - breaks[i - 1] > edge.eps * 2.0f) {
                emitSegment(edge, breaks[i - 1], breaks[i]);
            }
        }
    }
}



OctreeNodeLevel Octree::fetch(glm::vec3 pos, uint level, bool simplification, ThreadContext * context) const {
    glm::vec4 key = glm::vec4(pos, level);
    if(context->nodeCache.find(key) != context->nodeCache.end()) {
        return context->nodeCache[key];
    } else {
        OctreeNodeLevel nodeLevel = getNodeAt(pos, level, simplification);
        context->nodeCache[key] = nodeLevel;
        return nodeLevel;
    }
}



void Octree::expand(const ShapeArgs &args) {
    while (!args.function->isContained(*this, args.model, args.minSize)) {
        glm::vec3 point = args.function->getCenter(args.model);
        unsigned int i = getNodeIndex(point, *this) ^ 0x7;

        // Store old bounds before expansion for cube context
        BoundingCube oldRootCube(getMin(), getLengthX());

        setMin(getMin() - Octree::getShift(i) * getLengthX());
        setLength(getLengthX()*2);

        OctreeNode* oldRoot = root;
        OctreeNode* newRoot = allocator->allocate()->init(getCenter());
        ChildBlock* newBlock = newRoot->allocate(*allocator)->init();

        if (oldRoot != NULL) {
            ChildBlock* oldBlock = oldRoot->getBlock(*allocator);
            bool emptyNode = oldRoot->getType() == SpaceType::Empty;
            bool emptyBlock = (oldBlock == NULL || oldBlock->isEmpty());

            if (emptyNode && emptyBlock) {
                if (oldBlock != NULL) {
                    oldBlock = oldRoot->clear(*allocator, oldBlock, oldRootCube);
                }
                oldRoot = allocator->deallocate(oldRoot);
            }
        }
        if (newRoot == oldRoot) {
            throw std::runtime_error("Infinite recursion!");
        }

        if (oldRoot != NULL) {
            newBlock->set(i, oldRoot, *allocator);
        }
        root = newRoot;
    }
}

float Octree::evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> *cache, glm::vec3 p) const {
    auto it = cache->find(p);
    if (it != cache->end())
        return it->second;

    float d = args.function->distance(p, args.model);
    cache->try_emplace(p, d);
    return d;
}

void Octree::buildSDF(const ShapeArgs &args, OctreeNodeFrame &frame, float shapeSDF[8], float resultSDF[8], ThreadContext * threadContext) const {
    const glm::vec3 min = frame.cube.getMin();
    const glm::vec3 length = frame.cube.getLength();
    tsl::robin_map<glm::vec3, float> * shapeSdfCache = &threadContext->shapeSdfCache;

    for (uint i = 0; i < 8; ++i) {
        if(shapeSDF[i] == INFINITY) {
            shapeSDF[i] = evaluateSDF(args, shapeSdfCache, min + length * Octree::getShift(i));
        }
        if(resultSDF[i] == INFINITY) {
            resultSDF[i] = args.operation(frame.sdf[i], shapeSDF[i]);
        }
    }
}

void Octree::apply(
        float (*operation)(float, float),
        WrappedSignedDistanceFunction *function,
        const Transformation model,
        glm::vec4 translate,
        glm::vec4 scale,
        const TexturePainter &painter,
        float minSize,
        Simplifier &simplifier,
        const OctreeChangeHandler &changeHandler
    ) {
    threadsCreated = 0;
    *shapeCounter = 0;
    ShapeArgs args = ShapeArgs(operation, function, painter, model, translate, scale, simplifier, changeHandler, minSize);	
  	expand(args);
    OctreeNodeFrame frame = OctreeNodeFrame(root, *this, 0, root->sdf, DISCARD_BRUSH_INDEX, *this);
    ThreadContext localChunkContext = ThreadContext(*this);
    shape(frame, args, &localChunkContext, ContainmentType::Intersects);
    std::cout << "\t\tOctree::apply Ok! threads=" << threadsCreated << ", works=" << *shapeCounter << std::endl; 
}


SpaceType childToParent(bool childSolid, bool childEmpty) {
    if(childSolid) {
        return SpaceType::Solid;
    } else if(childEmpty) {
        return SpaceType::Empty;
    } else {
        return SpaceType::Surface;
    }
}

bool Octree::isChunkNode(float length) const {
    return chunkSize*0.5f < length && length <= chunkSize;
}

bool Octree::isThreadNode(float length, float minSize, int threadSize) const {
    return minSize*threadSize < length;
}

NodeOperationResult Octree::shape(OctreeNodeFrame frame, const ShapeArgs &args, ThreadContext * threadContext, ContainmentType parentContainment) {    
    ContainmentType check = parentContainment == ContainmentType::Intersects ? args.function->check(frame.cube, args.model, args.minSize) : parentContainment;
    OctreeNode * node = frame.node;
    bool process = true;
    auto isValidBrushIndex = [](int value) {
        return value > DISCARD_BRUSH_INDEX;
    };
    auto chooseBrushIndex = [&](int preferred, int fallback) {
        return isValidBrushIndex(preferred) ? preferred : fallback;
    };
    int sourceBrushIndex = chooseBrushIndex(node != NULL ? node->vertex.brushIndex : DISCARD_BRUSH_INDEX, frame.brushIndex);
    int brushIndex = sourceBrushIndex;

    if(check == ContainmentType::Disjoint) {
        process = false;
        SpaceType resultType = node ? node->getType() : SDF::eval(frame.sdf);
        return NodeOperationResult(node, SpaceType::Empty, INFINITY_ARRAY, resultType, frame.sdf, process, node ? node->isSimplified() : true, brushIndex);  // Skip this node
    }
    auto sameCube = [](const BoundingCube &a, const BoundingCube &b) {
        return a.getMin() == b.getMin() && a.getLengthX() == b.getLengthX();
    };

    float length = frame.cube.getLengthX();
    bool isChunk = isChunkNode(length) || sameCube(frame.cube, frame.chunkCube);
    bool isShapeLeaf = length <= args.minSize;
    bool isNodeLeaf = node == NULL || node->isLeaf();
    bool isLeaf = isShapeLeaf && isNodeLeaf;

    NodeOperationResult childResult[8];
    std::vector<std::thread> threads;
    threads.reserve(8);

    if (!isLeaf) {
        bool isChildThread = isThreadNode(length*0.5f, args.minSize, 16);
        bool isChildChunk = isChunkNode(length*0.5f);

        OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
        if(node != NULL) {
            node->getChildren(*allocator, children);
        }
        auto nearestChildBrushIndex = [&](uint targetIndex, int fallback) {
            if(isValidBrushIndex(fallback)) {
                return fallback;
            }

            int nearestBrushIndex = fallback;
            int nearestDistance = 4;
            for(uint j = 0; j < 8; ++j) {
                OctreeNode * candidate = children[j];
                if(candidate == NULL || !isValidBrushIndex(candidate->vertex.brushIndex)) {
                    continue;
                }

                uint bits = targetIndex ^ j;
                int distance = ((bits & 0x1) ? 1 : 0)
                    + ((bits & 0x2) ? 1 : 0)
                    + ((bits & 0x4) ? 1 : 0);
                if(distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestBrushIndex = candidate->vertex.brushIndex;
                    if(distance == 0) {
                        break;
                    }
                }
            }
            return nearestBrushIndex;
        };

        // Iterate nodes and spawn threads for child processing
        for (uint i = 0; i < 8; ++i) {
            OctreeNode * child = children[i];
            if(node!=NULL && child == node) {
                throw std::runtime_error("Infinite loop " + std::to_string((long)child) + " " + std::to_string((long)node));
            }

            float childSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
            int inheritedChildBrushIndex = nearestChildBrushIndex(i, sourceBrushIndex);
            int childBrushIndex = child != NULL
                ? chooseBrushIndex(child->vertex.brushIndex, inheritedChildBrushIndex)
                : inheritedChildBrushIndex;

            if(child != NULL) {
                SDF::copySDF(child->sdf, childSDF);
            } else {
                SDF::getChildSDF(frame.sdf, i, childSDF);
            }
            BoundingCube childCube = frame.cube.getChild(i);
            OctreeNodeFrame childFrame = OctreeNodeFrame(
                child, 
                childCube, 
                frame.level + 1, 
                childSDF,
                childBrushIndex,
                isChildChunk ? childCube : frame.chunkCube 
            );

            if(isChildThread) {
                ++threadsCreated;
                NodeOperationResult * result = &childResult[i];
                threads.emplace_back([this, childFrame, args, result, check]() {
                   ThreadContext localThreadContext(childFrame.cube);
                   *result = shape(childFrame, args, &localThreadContext, check);
                });
            } else {
                childResult[i] = shape(childFrame, args, threadContext, check);
            }

            (*shapeCounter)++;
        }
        for(std::thread &t : threads) {
            if(t.joinable()) t.join();
        }
    } 
 
    // Inherit SDFs from children
    float shapeSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    float resultSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    bool childResultSolid = true;
    bool childResultEmpty = true;
    bool childShapeSolid = true;
    bool childShapeEmpty = true;

    if(!isLeaf) {
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult & result = childResult[i];   
            childResultEmpty &= result.resultType == SpaceType::Empty;
            childResultSolid &= result.resultType == SpaceType::Solid;
            childShapeEmpty &= result.shapeType == SpaceType::Empty;
            childShapeSolid &= result.shapeType == SpaceType::Solid;
            // Disjoint children still carry valid existing/interpolated result SDF.
            if(result.resultSDF[i] != INFINITY) {
                resultSDF[i] = result.resultSDF[i];
            }
            if(result.process && result.shapeSDF[i] != INFINITY) {
                shapeSDF[i] = result.shapeSDF[i];
            } 
        }
    }

    // Build SDFs based on inheritance/execution
    buildSDF(args, frame, shapeSDF, resultSDF, threadContext);
    
    SpaceType shapeType = isLeaf ? SDF::eval(shapeSDF) : childToParent(childShapeSolid, childShapeEmpty);
    SpaceType resultType = isLeaf ? SDF::eval(resultSDF) : childToParent(childResultSolid, childResultEmpty);
    bool isResultSurface = resultType == SpaceType::Surface;
    bool isSimplified = isLeaf;

    if(shapeType == SpaceType::Empty && node != NULL) {
        process = false; 
    }
    else if(isResultSurface) {
        // Create nodes for surface results if they don't exist
        if(node == NULL) {
            node = allocator->allocate()->init(Vertex(frame.cube.getCenter()));   
        }

        if(node!= NULL) {
            node->vertex.position = SDF::getAveragePosition(resultSDF, frame.cube);
            node->vertex.normal = SDF::getNormalFromPosition(resultSDF, frame.cube, node->vertex.position);
            // Simplification & Painting
            if(isLeaf) {
                if(shapeType != SpaceType::Empty) {
                    isSimplified = true; 
                    brushIndex = args.painter.paint(node->vertex, args.translate, args.scale);
                }        
            } else {
                    if(isChunk) {
                        isSimplified = false;
                        brushIndex = DISCARD_BRUSH_INDEX;
                    } else {
                        SimplificationResult simplificationResult = args.simplifier.simplify(frame.chunkCube, frame.cube, resultSDF, childResult);
                        isSimplified = simplificationResult.isSimplified;
                        if(isSimplified) {
                            brushIndex = simplificationResult.brushIndex;
                        } 
                    }
                
                uint childNodes[8] = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX};
                for(uint i =0 ; i < 8 ; ++i) {
                    NodeOperationResult & child = childResult[i];
                    OctreeNode * childNode = child.node;
                    BoundingCube childCube = frame.cube.getChild(i);
                    bool childIsChunk = isChunkNode(childCube.getLengthX()) && sameCube(childCube, frame.chunkCube);
       
                    if(child.process) {
                        if(child.resultType != SpaceType::Surface) {
                            if(childNode == NULL) {
                                childNode = allocator->allocate()->init(Vertex(childCube.getCenter()));
                                childResult[i].node = childNode;
                            }
                            childNode->setType(child.resultType);
                            childNode->setSDF(child.resultSDF);
                            childNode->setSimplified(true);
                            childNode->setChunk(childIsChunk);
                            // Interpolated children inherit a real brush from the
                            // child result or nearest parent/source brush; chunk
                            // parents may deliberately carry DISCARD_BRUSH_INDEX.
                            int childNodeBrushIndex = chooseBrushIndex(child.brushIndex, chooseBrushIndex(brushIndex, sourceBrushIndex));
                            childNode->setBrush(childNodeBrushIndex);
                        }
                    }
                    childNodes[i] = allocator->nodeAllocator.getIndex(childNode);
                    if(frame.node != NULL && childNode == node) {
                        throw std::runtime_error("Infinite recursion! " + std::to_string((long) childNode) + " " + std::to_string((long)node) );
                    }                
                }
                node->setChildren(*allocator, childNodes);
            }
        }
    } 

    if(node!= NULL && process) {
        node->setSDF(resultSDF);
        node->setType(resultType);
        node->setChunk(isChunk);
        node->setSimplified(isSimplified);
        node->setBrush(brushIndex);
        ++node->version;

        if (isChunk) {
            OctreeNodeData nodeData(frame.level, node, frame.cube, check, nullptr);
            isResultSurface ? args.changeHandler.onNodeAdded(nodeData) : args.changeHandler.onNodeDeleted(nodeData);
        }
        if(!isResultSurface) {
            ChildBlock * block = node ? node->getBlock(*allocator) : NULL;
            if(block) {
                block = node->clear(*allocator, block, frame.cube);
            } 
        }
    }
    return NodeOperationResult(node, shapeType, shapeSDF, resultType, resultSDF, process, isSimplified, brushIndex);
}

void Octree::iterate(IteratorHandler &handler, OctreeNodeData data) {
	handler.iterate(*this, data);
}
void Octree::iterate(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, ContainmentType::Intersects, NULL);
	handler.iterate(*this, data);
}

void Octree::iterateFlat(IteratorHandler &handler, OctreeNodeData data) {
    handler.iterateFlatIn(*this, data);
}


void Octree::iterateFlat(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, ContainmentType::Intersects, NULL);
    handler.iterateFlatIn(*this, data);
}

void Octree::iterateParallel(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, ContainmentType::Intersects, NULL);
    handler.iterateBFS(*this, data);
    //handler.iterateParallelBFS(*this, data, threadPool);
}

void Octree::exportOctreeSerialization(OctreeSerialized * node) {
    std::cout << "exportOctreeSerialization()" << std::endl;
    for(int i = 0; i < 3; ++i) {
        node->min[i] = this->min[i];
        std::cout << "\tmin["<<std::to_string(i) <<"]: " << std::to_string(node->min[i]) << std::endl;
    }
    node->length = this->length;
    node->chunkSize = this->chunkSize;

    std::cout << "\tlength: " << std::to_string(node->length) << std::endl;
    std::cout << "\tchunkSize: " << std::to_string(node->chunkSize) << std::endl;
}

void Octree::exportNodesSerialization(std::vector<OctreeNodeCubeSerialized> * nodes) {
	std::cout << "exportNodesSerialization()" << std::endl;
    nodes->clear();
    nodes->reserve(10000000);
    int leafNodes = 0;
    root->exportSerialization(*allocator, nodes, &leafNodes, *this, *this, 0u);
	std::cout << "exportNodesSerialization Ok!" << std::endl;
}

void Octree::reset() {
    if(root != NULL) {
        allocator->childAllocator.reset();
        allocator->nodeAllocator.reset();
        this->root = allocator->allocate()->init(glm::vec3(getCenter()));
    }
}

void Octree::exportToJson(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Octree::exportToJson() Error opening file: " << filename << std::endl;
        return;
    }
    file.setf(std::ios::fixed);
    file << std::setprecision(6);

    std::function<void(const OctreeNode*, const BoundingCube&)> writeNode;
    writeNode = [&](const OctreeNode* node, const BoundingCube &cube) {
        if (node == NULL) {
            file << "null";
            return;
        }
        file << "{";
        const Vertex &v = node->vertex;
        file << "\"position\":[" << v.position.x << "," << v.position.y << "," << v.position.z << "],";
        file << "\"normal\":[" << v.normal.x << "," << v.normal.y << "," << v.normal.z << "],";
        file << "\"texCoord\":[" << v.texCoord.x << "," << v.texCoord.y << "],";
        file << "\"brushIndex\":" << v.brushIndex << ",";
        file << "\"bits\":" << (int)node->bits << ",";
        glm::vec3 min = cube.getMin();
        glm::vec3 len = cube.getLength();
        file << "\"min\":[" << min.x << "," << min.y << "," << min.z << "],";
        file << "\"length\":[" << len.x << "," << len.y << "," << len.z << "],";

        ChildBlock * block = node->getBlock(*allocator);
        if (block == NULL) {
            file << "\"children\": null";
        } else {
            OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            node->getChildren(*allocator, children);
            file << "\"children\":";
            file << "[";
            for (int i = 0; i < 8; ++i) {
                writeNode(children[i], cube.getChild(i));
                if (i < 7) file << ",";
            }
            file << "]";
        }
        file << "}";
    };

    file << "{ \"root\": ";
    writeNode(root, *this);
    file << " }\n";
    file.close();
    std::cout << "Octree exported to JSON: " << filename << std::endl;
}

void Octree::exportToBson(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Octree::exportToBson() Error opening file: " << filename << std::endl;
        return;
    }

    OctreeAllocator * alloc = this->allocator;

    auto appendInt32 = [](std::vector<uint8_t> &buf, int32_t v) {
        buf.push_back((uint8_t)(v & 0xff));
        buf.push_back((uint8_t)((v >> 8) & 0xff));
        buf.push_back((uint8_t)((v >> 16) & 0xff));
        buf.push_back((uint8_t)((v >> 24) & 0xff));
    };
    auto appendInt64 = [](std::vector<uint8_t> &buf, int64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)((v >> (8*i)) & 0xff));
    };
    auto appendDouble = [&](std::vector<uint8_t> &buf, double d) {
        uint64_t u = 0;
        std::memcpy(&u, &d, sizeof(double));
        for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)((u >> (8*i)) & 0xff));
    };
    auto appendCString = [&](std::vector<uint8_t> &buf, const std::string &s) {
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back(0x00);
    };

    std::function<std::vector<uint8_t>(const std::vector<double>&)> makeDoubleArrayDoc;
    makeDoubleArrayDoc = [&](const std::vector<double> &arr) -> std::vector<uint8_t> {
        std::vector<uint8_t> doc;
        appendInt32(doc, 0); // placeholder
        for (size_t i = 0; i < arr.size(); ++i) {
            doc.push_back(0x01); // double
            appendCString(doc, std::to_string(i));
            appendDouble(doc, arr[i]);
        }
        doc.push_back(0x00); // terminator
        int32_t size = (int32_t)doc.size();
        doc[0] = (uint8_t)(size & 0xff);
        doc[1] = (uint8_t)((size >> 8) & 0xff);
        doc[2] = (uint8_t)((size >> 16) & 0xff);
        doc[3] = (uint8_t)((size >> 24) & 0xff);
        return doc;
    };

    std::function<std::vector<uint8_t>(const OctreeNode*, const BoundingCube&)> buildNodeDoc;
    buildNodeDoc = [&](const OctreeNode* node, const BoundingCube &cube) -> std::vector<uint8_t> {
        std::vector<uint8_t> doc;
        appendInt32(doc, 0); // placeholder

        if (node == nullptr) {
            // empty document
            doc.push_back(0x00);
            int32_t size = (int32_t)doc.size();
            doc[0] = (uint8_t)(size & 0xff);
            doc[1] = (uint8_t)((size >> 8) & 0xff);
            doc[2] = (uint8_t)((size >> 16) & 0xff);
            doc[3] = (uint8_t)((size >> 24) & 0xff);
            return doc;
        }

        // position (array)
        std::vector<double> pos = { node->vertex.position.x, node->vertex.position.y, node->vertex.position.z };
        std::vector<uint8_t> posArr = makeDoubleArrayDoc(pos);
        doc.push_back(0x04); appendCString(doc, "position"); doc.insert(doc.end(), posArr.begin(), posArr.end());

        // normal
        std::vector<double> nrm = { node->vertex.normal.x, node->vertex.normal.y, node->vertex.normal.z };
        std::vector<uint8_t> nrmArr = makeDoubleArrayDoc(nrm);
        doc.push_back(0x04); appendCString(doc, "normal"); doc.insert(doc.end(), nrmArr.begin(), nrmArr.end());

        // texCoord
        std::vector<double> tex = { node->vertex.texCoord.x, node->vertex.texCoord.y };
        std::vector<uint8_t> texArr = makeDoubleArrayDoc(tex);
        doc.push_back(0x04); appendCString(doc, "texCoord"); doc.insert(doc.end(), texArr.begin(), texArr.end());

        // brushIndex (int32)
        doc.push_back(0x10); appendCString(doc, "brushIndex"); appendInt32(doc, node->vertex.brushIndex);

        // bits (int32)
        doc.push_back(0x10); appendCString(doc, "bits"); appendInt32(doc, (int32_t)node->bits);

        // min
        glm::vec3 minv = cube.getMin();
        std::vector<double> minArr = { minv.x, minv.y, minv.z };
        std::vector<uint8_t> minDoc = makeDoubleArrayDoc(minArr);
        doc.push_back(0x04); appendCString(doc, "min"); doc.insert(doc.end(), minDoc.begin(), minDoc.end());

        // length
        glm::vec3 lenv = cube.getLength();
        std::vector<double> lenArr = { lenv.x, lenv.y, lenv.z };
        std::vector<uint8_t> lenDoc = makeDoubleArrayDoc(lenArr);
        doc.push_back(0x04); appendCString(doc, "length"); doc.insert(doc.end(), lenDoc.begin(), lenDoc.end());

        // children
        ChildBlock * block = node->getBlock(*alloc);
        if (block == NULL) {
            doc.push_back(0x0A); appendCString(doc, "children"); // null
        } else {
            OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            node->getChildren(*alloc, children);
            std::vector<uint8_t> arr;
            appendInt32(arr, 0); // placeholder
            for (int i = 0; i < 8; ++i) {
                OctreeNode *child = children[i];
                std::string key = std::to_string(i);
                if (child == NULL) {
                    arr.push_back(0x0A); appendCString(arr, key); // null
                } else {
                    std::vector<uint8_t> childDoc = buildNodeDoc(child, cube.getChild(i));
                    arr.push_back(0x03); appendCString(arr, key); // embedded document
                    arr.insert(arr.end(), childDoc.begin(), childDoc.end());
                }
            }
            arr.push_back(0x00);
            int32_t arrSize = (int32_t)arr.size();
            arr[0] = (uint8_t)(arrSize & 0xff);
            arr[1] = (uint8_t)((arrSize >> 8) & 0xff);
            arr[2] = (uint8_t)((arrSize >> 16) & 0xff);
            arr[3] = (uint8_t)((arrSize >> 24) & 0xff);

            doc.push_back(0x04); appendCString(doc, "children"); doc.insert(doc.end(), arr.begin(), arr.end());
        }

        doc.push_back(0x00);
        int32_t size = (int32_t)doc.size();
        doc[0] = (uint8_t)(size & 0xff);
        doc[1] = (uint8_t)((size >> 8) & 0xff);
        doc[2] = (uint8_t)((size >> 16) & 0xff);
        doc[3] = (uint8_t)((size >> 24) & 0xff);
        return doc;
    };

    // top-level document
    std::vector<uint8_t> top;
    appendInt32(top, 0);
    if (root == NULL) {
        top.push_back(0x0A); appendCString(top, "root");
    } else {
        std::vector<uint8_t> rootDoc = buildNodeDoc(root, *this);
        top.push_back(0x03); appendCString(top, "root"); top.insert(top.end(), rootDoc.begin(), rootDoc.end());
    }
    top.push_back(0x00);
    int32_t topSize = (int32_t)top.size();
    top[0] = (uint8_t)(topSize & 0xff);
    top[1] = (uint8_t)((topSize >> 8) & 0xff);
    top[2] = (uint8_t)((topSize >> 16) & 0xff);
    top[3] = (uint8_t)((topSize >> 24) & 0xff);

    file.write(reinterpret_cast<const char*>(top.data()), top.size());
    file.close();
    std::cout << "Octree exported to BSON: " << filename << std::endl;
}
