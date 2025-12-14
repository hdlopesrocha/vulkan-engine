#ifndef SPACE_ITERATOR_HPP
#define SPACE_ITERATOR_HPP

#include "types.hpp"
#include "node.hpp"
#include "octree.hpp"
#include "ThreadPool.hpp"
#include <stack>

template <typename T> class GeometryBuilder {
public:
    virtual InstanceGeometry<T> * build(Octree * tree, OctreeNodeData &params, ThreadContext * context) = 0;
};

class IteratorHandler {
    std::stack<OctreeNodeData> flatData;
    std::stack<StackFrame> stack;
    std::stack<StackFrameOut> stackOut;

public:
    virtual bool test(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void before(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void after(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) = 0;
    void iterate(const Octree &tree, OctreeNodeData &params);
    void iterateMultiThreaded(const Octree &tree, OctreeNodeData &params);

    void iterateFlatIn(const Octree &tree, OctreeNodeData &params);
    void iterateFlatOut(const Octree &tree, OctreeNodeData &params);
    void iterateFlat(const Octree &tree, OctreeNodeData &params);
    void iterateBFS(const Octree &tree, OctreeNodeData &rootParams);
    void iterateParallelBFS(const Octree &tree, OctreeNodeData &rootParams, ThreadPool& pool);
};

template <typename T> class InstanceBuilderHandler {
public:
    virtual void handle(const Octree &tree, OctreeNodeData &data, std::vector<T> * instances, ThreadContext * context) = 0;
};

template <typename T> class InstanceBuilder : public IteratorHandler{
    InstanceBuilderHandler<T> * handler;
    std::vector<T> * instances;
    ThreadContext * context;
public:
    InstanceBuilder(InstanceBuilderHandler<T> * handler, std::vector<T> * instances, ThreadContext * context);
    void before(const Octree &tree, OctreeNodeData &params);
    void after(const Octree &tree, OctreeNodeData &params);
    bool test(const Octree &tree, OctreeNodeData &params);
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]);
};

class Tesselator : public OctreeNodeTriangleHandler{
    ThreadContext * context;
public:
    Geometry * geometry;
    Tesselator(long * count, ThreadContext * context);
    void handle(Vertex &v0, Vertex &v1, Vertex &v2, bool sign) override;
};

class Processor : public IteratorHandler {
    ThreadPool &threadPool;
    ThreadContext * context;
    std::vector<OctreeNodeTriangleHandler*> * handlers;
    std::unordered_set<BoundingCube,BoundingCubeHasher> iteratedCubes;

public:
    Processor(long * count, ThreadPool &threadPool, ThreadContext * context, std::vector<OctreeNodeTriangleHandler*> * handlers);
    void iterate(const Octree &tree, OctreeNodeData &params);
    void before(const Octree &tree, OctreeNodeData &params) override;
    void after(const Octree &tree, OctreeNodeData &params) override;
    bool test(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
    void virtualize(Octree * tree, const BoundingCube &cube, float * sdf, uint level, uint levels);
};

class OctreeFile {
    Octree * tree;
    std::string filename;
public:
    OctreeFile(Octree * tree, std::string filename);
    void save(std::string baseFolder, float chunkSize);
    void load(std::string baseFolder, float chunkSize);
    AbstractBoundingBox& getBox();
    OctreeNode * loadRecursive(int i, std::vector<OctreeNodeSerialized> * nodes, float chunkSize, std::string filename, BoundingCube cube, std::string baseFolder);
    uint saveRecursive(OctreeNode * node, std::vector<OctreeNodeSerialized> * nodes, float chunkSize, std::string filename, BoundingCube cube, std::string baseFolder);
};

class OctreeNodeFile {
    OctreeNode * node;
    std::string filename;
    Octree * tree;
public:
    OctreeNodeFile(Octree * tree, OctreeNode * node, std::string filename);
    void save(std::string baseFolder);
    void load(std::string baseFolder, BoundingCube &cube);
    OctreeNode * loadRecursive(OctreeNode * node, int i, BoundingCube &cube, std::vector<OctreeNodeSerialized> * nodes);
    uint saveRecursive(OctreeNode * node, std::vector<OctreeNodeSerialized> * nodes);
};

class OctreeVisibilityChecker : public IteratorHandler{
    Frustum frustum;
    glm::vec3 viewDir;
public:
    glm::vec3 sortPosition;
    std::vector<OctreeNodeData> visibleNodes;
    std::mutex mutex;
    OctreeVisibilityChecker();
    void update(glm::mat4 m);
    void before(const Octree &tree, OctreeNodeData &params) override;
    void after(const Octree &tree, OctreeNodeData &params) override;
    bool test(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
};

#endif
