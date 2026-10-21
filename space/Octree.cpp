#include "Octree.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include "../math/BrushMode.hpp"
#include "NodeOperationResult.hpp"
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <set>
#include <array>
#include "OctreeAllocator.hpp"
#include "OctreeNode.hpp"
#include "OctreeNodeData.hpp"
#include "OctreeSharedLock.hpp"
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
Octree::Octree(const BoundingCube &minCube, float chunkSize_) : BoundingCube(minCube), allocator(new OctreeAllocator()) {
    this->chunkSize = chunkSize_;
	this->root = allocator->allocate()->init(glm::vec3(minCube.getCenter()));
	initialize();
}

Octree::Octree() : Octree(glm::vec3(0.0f), 1.0f) {
    this->chunkSize = 1.0f;
    this->root = NULL;
    initialize();
}

int getNodeIndex(const glm::vec3 &vec, const BoundingCube &cube) {
    return cube.getChildIndex(vec);
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

bool Octree::intersect(const Ray& ray, glm::vec3& outPos) const {
    float tNear, tFar;
    if (!ray.intersects(*this, &tNear, &tFar))
        return false;

    struct Entry { OctreeNode* node; BoundingCube cube; float tNear; };
    std::vector<Entry> stack;
    stack.push_back({root, *this, tNear});

    float bestT = tFar;
    bool found = false;

    while (!stack.empty()) {
        Entry e = stack.back();
        stack.pop_back();

        if (e.tNear >= bestT) continue;

        if (e.node->isLeaf()) {
            if (e.node->getType() == SpaceType::Surface) {
                float tn, tf;
                if (ray.intersects(e.cube, &tn, &tf) && tn < bestT) {
                    bestT = tn;
                    found = true;
                }
            }
            continue;
        }

        ChildBlock* block = e.node->getBlock(*allocator);
        if (!block) continue;

        struct ChildE { int idx; float tNear; };
        std::vector<ChildE> children;
        for (int i = 0; i < 8; ++i) {
            OctreeNode* child = block->get(i, *allocator);
            if (!child) continue;
            BoundingCube childCube = e.cube.getChild(i);
            float tn, tf;
            if (ray.intersects(childCube, &tn, &tf) && tf >= 0.0f && tn < bestT) {
                children.push_back({i, tn});
            }
        }

        std::sort(children.begin(), children.end(),
                  [](const ChildE& a, const ChildE& b) { return a.tNear < b.tNear; });
        for (int i = static_cast<int>(children.size()) - 1; i >= 0; --i) {
            int idx = children[i].idx;
            OctreeNode* child = block->get(idx, *allocator);
            stack.push_back({child, e.cube.getChild(idx), children[i].tNear});
        }
    }

    if (found) {
        outPos = ray.pointAt(bestT);
        return true;
    }
    return false;
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
        if (simplification && node->getLod() == 1u) {
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
        if (simplification && node->getLod() == 1u) {
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
    return INFINITY;
}




OctreeNodeLevel Octree::fetch(glm::vec3 pos, uint level, bool simplification, OctreeThreadContext * context) const {
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
    while (!args.function.isContained(*this)) {
        glm::vec3 point = args.function.getCenter();
        unsigned int i = getNodeIndex(point, *this) ^ 0x7;

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
                    oldBlock = oldRoot->clear(*allocator, oldBlock);
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

    float d = args.function.distance(p);
    cache->try_emplace(p, d);
    return d;
}

void Octree::buildShapeSDF(const bool isLeaf,const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext, bool force) const {
    const glm::vec3 cubeMin = frame.cube.getMin();
    const glm::vec3 cubeLength = frame.cube.getLength();
    tsl::robin_map<glm::vec3, float> * shapeSdfCache = &threadContext->shapeSdfCache;

    if(isLeaf || force) {
        for (uint i = 0; i < 8; ++i) {
            r.shapeSDF[i] = evaluateSDF(args, shapeSdfCache, cubeMin + cubeLength * Octree::getShift(i));
        }
        r.shapeType = SDF::eval(r.shapeSDF);
    } else {
        bool childShapeSolid = true;
        bool childShapeEmpty = true;
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult &child = children[i];
            childShapeEmpty &= child.shapeType == SpaceType::Empty;
            childShapeSolid &= child.shapeType == SpaceType::Solid;
            r.shapeSDF[i] = child.shapeSDF[i];
        }
        r.shapeType = childToParent(childShapeSolid, childShapeEmpty);
    }
}

void Octree::buildResultSDF(const bool isLeaf, const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext) const {
    if(isLeaf) {
        for (uint i = 0; i < 8; ++i) {
            r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
        }
        r.resultType = SDF::eval(r.resultSDF);
    } else {
        bool childResultSolid = true;
        bool childResultEmpty = true;
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult &child = children[i];
            childResultEmpty &= child.resultType == SpaceType::Empty;
            childResultSolid &= child.resultType == SpaceType::Solid;
            r.resultSDF[i] = child.resultSDF[i];
        }
        r.resultType = childToParent(childResultSolid, childResultEmpty);
    }
}

void Octree::apply(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const Transformation &model,
        const TexturePositionPainter &painter,
        const TextureCoordPainter &coordPainter,
        float minSize,
        const Simplifier &simplifier,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    ) {
    std::unique_lock<std::shared_mutex> writeLock(treeMutex);

    ShapeArgs args = ShapeArgs(operation, function, painter, coordPainter, model, simplifier, minSize);	
    expand(args);
    OctreeNodeFrame frame = OctreeNodeFrame(root, NULL, *this, root ? root->getType() : SpaceType::Empty, 0, root ? root->sdf : nullptr, DISCARD_BRUSH_INDEX, *this);
    OctreeThreadContext localChunkContext;
    NodeOperationResult r = NodeOperationResult();
    shape(r, frame, args, &localChunkContext, updateHandler, deleteHandler);
}

int Octree::heightRootToChunk(int lod, float minSize) const {
    // Number of subdivision levels a chunk-size node can hold above minSize.
    // Guarded against degenerate configurations (chunkSize < minSize).
    const float ratio = std::max(1.0f, chunkSize / std::max(minSize, 1.0f));
    const int maxLevels = static_cast<int>(std::floor(std::log2(ratio)));
    return maxLevels - lod;
}


bool Octree::isChunkNode(float nodeLength) const {
    return chunkSize*0.5f < nodeLength && nodeLength <= chunkSize;
}

bool Octree::isThreadNode(float nodeLength, float minSize, int threadSize) const {
    return minSize*threadSize*0.5f < nodeLength && nodeLength <= minSize*threadSize;
}

void Octree::shapeChildren(
    const OctreeNodeFrame &frame, 
    const ShapeArgs &args, 
    OctreeThreadContext * threadContext, 
    NodeOperationResult childResult[8],
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler) {
    float childLength = frame.cube.getLengthX()*0.5f;
    bool isChildThread = isThreadNode(childLength, args.minSize, 16);
    bool isChildChunk = isChunkNode(childLength);
    std::vector<std::future<void>> futures;

    OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    OctreeNode * node = frame.node;
    if(node != NULL) {
        node->getChildren(*allocator, children);
    }

    int brushIndex = node ? node->brushIndex : frame.brushIndex;
    if(brushIndex == DISCARD_BRUSH_INDEX) {
        brushIndex = frame.brushIndex;
    }
    glm::vec3 hsv = node ? node->hsv : frame.hsv;

    // Iterate nodes and submit threaded children to the pool
    for (uint i = 0; i < 8; ++i) {
        OctreeNode * child = children[i];
        if(node!=NULL && child == node) {
            throw std::runtime_error("Infinite loop " + std::to_string((long)child) + " " + std::to_string((long)node));
        }

        float childSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
        int childBrushIndex = child ? child->brushIndex : brushIndex;
        if(childBrushIndex == DISCARD_BRUSH_INDEX) {
            childBrushIndex = brushIndex;
        }
        glm::vec3 childHsv = child ? child->hsv : hsv;

        if(child != NULL) {
            SDF::copySDF(child->sdf, childSDF);
        } else {
            SDF::getChildSDF(frame.sdf, i, childSDF);
        }
        BoundingCube childCube = frame.cube.getChild(i);
        OctreeNodeFrame childFrame = OctreeNodeFrame(
            child,
            child ? frame.iteratedNode : frame.node,
            childCube,
            child ? child->getType() : frame.type,
            frame.level + 1,
            childSDF,
            childBrushIndex,
            isChildChunk ? childCube : frame.chunkCube
        );
        childFrame.hsv = childHsv;

    
        if(isChildThread) {
            NodeOperationResult * result = &childResult[i];
            inFlightShapeOps.fetch_add(1);
            futures.push_back(threadPool.enqueue([this, childFrame, args, result, &updateHandler, &deleteHandler]() {
                OctreeThreadContext localOctreeThreadContext;
                shape(*result, childFrame, args, &localOctreeThreadContext, updateHandler, deleteHandler);
                inFlightShapeOps.fetch_sub(1);
            }));
        } else {
            shape(childResult[i], childFrame, args, threadContext, updateHandler, deleteHandler);
        }
    
    }
    if(isChildThread) {
        for(auto &f : futures) {
            f.wait();
        }
    }
}


void Octree::shape(
    NodeOperationResult &r,
    OctreeNodeFrame frame, 
    const ShapeArgs &args, 
    OctreeThreadContext * threadContext,
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler
) {    
    r.node = frame.node;
    const float nodeLength = frame.cube.getLengthX();
    const bool isShapeLeaf = nodeLength <= args.minSize;
    const bool isNodeLeaf = r.node == NULL || r.node->isLeaf();
    r.brushIndex = r.node ? r.node->brushIndex : frame.brushIndex;
    r.brushHsv = r.node ? r.node->hsv : frame.hsv;
    bool isChunk = isChunkNode(nodeLength);
    bool isLeaf = isShapeLeaf && isNodeLeaf;
    r.selectedLod = isLeaf ? 1u : 0u;
    r.selectedChunkLod = isChunk ? 1u : 0u;

    NodeOperationResult children[8] = { 
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult(),
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult() 
    };

    buildShapeSDF(isLeaf, args, frame, r, children, threadContext, true);

    const glm::vec3 center = frame.cube.getCenter();
    float shapeSdfCenter = evaluateSDF(args, &threadContext->shapeSdfCache, center);

    bool process = true;
    bool processed = false;
    if(!isLeaf) {
        const float halfDiagonal = nodeLength * 0.866025403784439f;

        if(!processed && isNodeLeaf) {
            bool allInfinity = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] != INFINITY) { allInfinity = false; break; }
            if(allInfinity) {
                if(args.operation->propagatesFromInfinity()) {
                    if(shapeSdfCenter < -halfDiagonal) {
                        SDF::copySDF(r.shapeSDF, r.resultSDF);
                        r.shapeType = SpaceType::Solid;
                        r.resultType = SpaceType::Solid;
                        r.selectedLod = 0;
                        processed = true;
                    } else if(shapeSdfCenter > halfDiagonal) {
                        SDF::copySDF(r.shapeSDF, r.resultSDF);
                        r.shapeType = SpaceType::Empty;
                        r.resultType = SpaceType::Empty;
                        r.selectedLod = 0;
                        processed = true;
                    }
                } else {
                    r.shapeType = SDF::eval(r.shapeSDF);
                    r.resultType = SpaceType::Empty;
                    r.selectedLod = 0;
                    processed = true;
                }
            }
            
        }

        if(!processed && frame.type == SpaceType::Solid &&
           args.operation->preservesSolid()) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {

            if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                if(r.node != NULL) {
                    glm::vec3 _pos = SDF::getPosition(r.resultSDF, frame.cube);
                    glm::vec3 _norm = SDF::getNormalFromPosition(r.resultSDF, frame.cube, _pos);
                    Vertex _v(_pos, _norm, glm::vec2(0.0f), r.node->brushIndex);
                    _v.hsv = r.node->hsv;
                    r.brushIndex = args.painter.paint(_v);
                    r.brushHsv = args.painter.paintHSV(_v);
                    r.textureType = static_cast<int>(args.coordPainter.textureType());
                    r.data = args.coordPainter.paintUV(_v);
                    r.node->hsv = r.brushHsv;
                    r.node->brushIndex = r.brushIndex;
                    r.node->textureType = r.textureType;
                    r.node->data = r.data;
                    r.node->setBrush(r.brushIndex);
                }
            }
            for(uint i = 0; i < 8; ++i) {
                r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
            }
            r.resultType = SpaceType::Solid;
            processed = true;
            }
        }

        if(!processed && frame.type == SpaceType::Empty && args.operation->preservesEmpty()) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = SpaceType::Empty;
                r.brushIndex = frame.brushIndex;
                r.brushHsv = frame.hsv;
                processed = true;
            }
        }

        if(!processed && frame.type == SpaceType::Solid) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {

            const float existingSdfCenter = SDF::interpolate(frame.sdf, center, frame.cube);
            const float resultSdfCenter = args.operation->combine(existingSdfCenter, shapeSdfCenter);
            if(resultSdfCenter < -halfDiagonal) {
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = SpaceType::Solid;
                if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                    if(r.node != NULL) {
                        glm::vec3 _pos = SDF::getPosition(r.resultSDF, frame.cube);
                        glm::vec3 _norm = SDF::getNormalFromPosition(r.resultSDF, frame.cube, _pos);
                        Vertex _v(_pos, _norm, glm::vec2(0.0f), r.node->brushIndex);
                        _v.hsv = r.node->hsv;
                        r.brushIndex = args.painter.paint(_v);
                        r.brushHsv = args.painter.paintHSV(_v);
                        r.textureType = static_cast<int>(args.coordPainter.textureType());
                        r.data = args.coordPainter.paintUV(_v);
                        r.node->hsv = r.brushHsv;
                        r.node->brushIndex = r.brushIndex;
                        r.node->textureType = r.textureType;
                        r.node->data = r.data;
                        r.node->setBrush(r.brushIndex);
                    }
                }
                processed = true;
            }
            }
            
        }

        if(!processed) {
            process = shapeSdfCenter <= halfDiagonal;

            if(process) {
                const ContainmentType check = args.function.check(frame.cube);
                process = check != ContainmentType::Disjoint;
            }
            if(!process && frame.node == NULL && frame.type == SpaceType::Surface) {
                process = true;
            }
            if(process) {    
                shapeChildren(frame, args, threadContext, children, updateHandler, deleteHandler);
            } else {
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = frame.type;
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
            }
        }
    }

    if(isLeaf || (process && !processed && !isLeaf)) {
        buildShapeSDF(isLeaf, args, frame, r, children, threadContext, false);
        buildResultSDF(isLeaf, args, frame, r, children, threadContext);
    }

    bool interpolatedSurface = (frame.node == NULL)
                                && SDF::eval(frame.sdf) == SpaceType::Surface 
                                && frame.type == SpaceType::Surface
                                ;


    if(r.shapeType != SpaceType::Empty || interpolatedSurface || r.node != NULL) {
        if(r.resultType == SpaceType::Surface) {
            // Create nodes for surface results if they don't exist
            if(r.node == NULL) {
                r.node = allocator->allocate()->init(Vertex(frame.cube.getCenter()));   
            }

            if(r.node!= NULL) {
                glm::vec3 _pos = SDF::getPosition(r.resultSDF, frame.cube);
                glm::vec3 _norm = SDF::getNormalFromPosition(r.resultSDF, frame.cube, _pos);
                // Simplification & Painting
                if(isLeaf) {
                    if(r.shapeType != SpaceType::Empty) {
                        Vertex _v(_pos, _norm, glm::vec2(0.0f), r.node->brushIndex);
                        _v.hsv = r.node->hsv;
                        r.brushIndex = args.painter.paint(_v);
                        r.node->hsv = args.painter.paintHSV(_v);
                        r.node->brushIndex = r.brushIndex;
                        r.node->textureType = static_cast<int>(args.coordPainter.textureType());
                        r.node->data = args.coordPainter.paintUV(_v);
                        r.textureType = r.node->textureType;
                        r.data = r.node->data;
                        r.brushHsv = r.node->hsv;
                    }  
                } else  {
                
                    if(process) {
                        if (!isChunk) {
                            // Pass frame.chunkCube so the simplifier can guard chunk borders.
                            SimplificationResult simplificationResult = args.simplifier.simplify(frame.cube, r.resultSDF, children, frame.chunkCube);
                            if(simplificationResult.isSimplified) {
                                r.selectedLod = 1u;
                            }
                        }
                        OctreeNode * childNodes[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
                        for(uint i =0 ; i < 8 ; ++i) {
                            NodeOperationResult &child = children[i];
                            OctreeNode * childNode = child.node;

                            if(child.resultType != SpaceType::Surface) {
                                if(childNode == NULL) {
                                    BoundingCube childCube = frame.cube.getChild(i);
                                    childNode = allocator->allocate()->init(Vertex(childCube.getCenter()));
                                    children[i].node = childNode;
                                }
                                childNode->setType(child.resultType);
                                childNode->setSDF(child.resultSDF);
                                childNode->setChunk(false);
                                childNode->setBrush(r.brushIndex);
                                childNode->textureType = r.textureType;
                                childNode->data = r.data;
                                childNode->setLod(child.selectedLod);
                                childNode->setChunkLod(child.selectedChunkLod);
                                childNode->hsv = child.brushHsv;
                            }

                            if(frame.node != NULL && childNode == r.node) {
                                throw std::runtime_error("Infinite recursion! " + std::to_string((long) childNode) + " " + std::to_string((long)r.node) );
                            }        
                            childNodes[i] = childNode;
                        }
                        r.node->setChildren(*allocator, childNodes);
                    }
                }
            }
        }
    }

    if(!isLeaf) {
        uint8_t selectedLod = 0;
        uint8_t selectedChunkLod = 0;

        for(uint i =0 ; i < 8 ; ++i) {
            NodeOperationResult &child = children[i];
            OctreeNode * childNode = child.node;
            std::unordered_map<int, int> brushCounts;
            int bestCount = 0;
            if(childNode != NULL && childNode->getType() == SpaceType::Surface) {
                const uint8_t childLod = childNode->getLod();
                const uint8_t childChunkLod = childNode->getChunkLod();
                if(childLod > 0u) {
                    selectedLod = selectedLod == 0u ?
                        childLod : glm::min(selectedLod, childLod);
                }
                if(childChunkLod > 0u) {
                    selectedChunkLod = selectedChunkLod == 0u ?
                        childChunkLod : glm::min(selectedChunkLod, childChunkLod);
                }
                const int childBrush = childNode->getBrush();
                if(childBrush > DISCARD_BRUSH_INDEX) {
                    const int count = ++brushCounts[childBrush];
                    if(count > bestCount) {
                        bestCount = count;
                        r.brushIndex = childBrush;
                        r.brushHsv = childNode->hsv;
                        r.textureType = childNode->textureType;
                        r.data = childNode->data;
                    }
                }
            }    
        }
        if(selectedLod > 0u) {
            r.selectedLod = selectedLod + 1u;
        }
        if(selectedChunkLod > 0u) {
            r.selectedChunkLod = selectedChunkLod + 1u;
        }
    }

    // Materialize this node's properties for both subdivided and terminal nodes.
    if(process && r.node != NULL) {
        if(r.resultType != SpaceType::Surface) {
            r.node->clear(*allocator, NULL);
        }
        r.node->setType(r.resultType);
        r.node->setSDF(r.resultSDF);
        r.node->setChunk(isChunk);
        r.node->setLod(r.selectedLod);
        r.node->setChunkLod(r.selectedChunkLod);
        r.node->setBrush(r.brushIndex);
        r.node->hsv = r.brushHsv;

        if(r.node->getChunkLod() > 0 && r.resultType == SpaceType::Surface) {
            ++r.node->version;
            OctreeNodeData data = OctreeNodeData(frame.level, r.node, frame.cube, nullptr);
            updateHandler(data);
        }
    }
}

Octree::~Octree() {
    while (inFlightShapeOps.load() > 0) {
        std::this_thread::yield();
    }
    threadPool.stop();
    delete allocator;
}

void Octree::reset() {
    while (inFlightShapeOps.load() > 0) {
        std::this_thread::yield();
    }
    if(root != NULL) {
        allocator->childAllocator.reset();
        allocator->nodeAllocator.reset();
        this->root = allocator->allocate()->init(glm::vec3(getCenter()));
    }
}

