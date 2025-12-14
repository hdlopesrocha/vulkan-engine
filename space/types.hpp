#ifndef SPACE_TYPES_HPP
#define SPACE_TYPES_HPP

#include <vector>
#include <tsl/robin_map.h>
#include <mutex>
#include <functional>
#include "../math/math.hpp"
#include "../sdf/SDF.hpp"

const float INFINITY_ARRAY [8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
const uint UINT_MAX_ARRAY [8] = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX};

typedef std::function<void(const BoundingCube &childCube, const float sdf[8], uint level)> IterateBorderHandler;

#pragma pack(16)  // Ensure 16-byte alignment for UBO
struct OctreeSerialized {
    public:
    glm::vec3 min;
    float length;
    float chunkSize;
};
#pragma pack()  // Reset to default packing

struct alignas(16) OctreeNodeCubeSerialized {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    int brushIndex;
    uint children[8];
    glm::vec3 min;
    uint bits;
    glm::vec3 length;
    uint level;

    OctreeNodeCubeSerialized();
    OctreeNodeCubeSerialized(float * sdf, BoundingCube cube, Vertex vertex, uint bits, uint level);
};

#pragma pack(16)
struct OctreeNodeSerialized {
    public:
    float sdf[8];
    uint children[8] = {0,0,0,0,0,0,0,0};
    int brushIndex;
    uint8_t bits;
};
#pragma pack()

class OctreeNode;
class OctreeAllocator;
class Simplifier;
class OctreeChangeHandler {
public:
    virtual void create(OctreeNode* nodeId) = 0;
    virtual void update(OctreeNode* nodeId) = 0;
    virtual void erase(OctreeNode* nodeId) = 0;
};
class TexturePainter;

class OctreeNodeLevel {
public:
    OctreeNode* node;
    uint level;
    OctreeNodeLevel() : node(NULL), level(0) {}
    OctreeNodeLevel(OctreeNode* node, uint level) : node(node), level(level) {}
};

struct OctreeNodeData {
public:
    uint level;
    OctreeNode * node;
    BoundingCube cube;
    ContainmentType containmentType;
    void * context;
    float sdf[8];
    OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context, float * sdf);
    OctreeNodeData(const OctreeNodeData &data);
    OctreeNodeData();
};

class OctreeNodeTriangleHandler {
public:
    long * count;
    OctreeNodeTriangleHandler(long * count);
    virtual void handle(Vertex &v0, Vertex &v1, Vertex &v2, bool sign) = 0;
};

struct OctreeNodeFrame {
    OctreeNode* node;
    BoundingCube cube;
    uint level;
    float sdf[8];
    int brushIndex;
    bool interpolated;
    BoundingCube chunkCube;
    OctreeNodeFrame();
    OctreeNodeFrame(const OctreeNodeFrame &t);
    OctreeNodeFrame(OctreeNode* node, BoundingCube cube, uint level, float * sdf, int brushIndex, bool interpolated, BoundingCube chunkCube);
};

struct StackFrame : public OctreeNodeData {
    uint8_t childIndex;
    uint8_t internalOrder[8];
    bool secondVisit;

    StackFrame(const OctreeNodeData &data, uint8_t childIndex, bool secondVisit) : OctreeNodeData(data) {
        this->childIndex = childIndex;
        this->secondVisit = secondVisit;
    }
};

struct StackFrameOut : public OctreeNodeData {
    bool visited;
    StackFrameOut(const OctreeNodeData &data, bool visited) : OctreeNodeData(data) {
        this->visited = visited;
    }
};

struct NodeOperationResult {
    OctreeNode * node;
    SpaceType shapeType;
    SpaceType resultType;
    bool process;
    float resultSDF[8];
    float shapeSDF[8];
    bool isSimplified;
    int brushIndex;
    NodeOperationResult();
    NodeOperationResult(OctreeNode * node, SpaceType shapeType, SpaceType resultType, const float * resultSDF, const float * shapeSDF, bool process, bool isSimplified, int brushIndex);
};

struct ShapeArgs {
    float (*operation)(float, float);
    WrappedSignedDistanceFunction * function;
    const TexturePainter &painter;
    const Transformation model;
    glm::vec4 translate;
    glm::vec4 scale;
    Simplifier &simplifier;
    OctreeChangeHandler * changeHandler;
    float minSize;

    ShapeArgs(
        float (*operation)(float, float),
        WrappedSignedDistanceFunction * function,
        const TexturePainter &painter,
        const Transformation model,
        glm::vec4 translate,
        glm::vec4 scale,
        Simplifier &simplifier,
        OctreeChangeHandler * changeHandler,
        float minSize
    );
};

class ThreadContext {
public:
    tsl::robin_map<glm::vec3, float> shapeSdfCache;
    tsl::robin_map<glm::vec4, OctreeNodeLevel> nodeCache;
    std::shared_mutex mutex;
    BoundingCube cube;
    ThreadContext(BoundingCube cube);
};

struct alignas(16) InstanceData {
public:
    glm::mat4 matrix;
    float shift;
    uint animation;
    InstanceData();
    InstanceData(uint animation, const glm::mat4 &matrix, float shift);
};

struct DebugInstanceData {
public:
    glm::vec4 sdf1;
    glm::vec4 sdf2;
    glm::mat4 matrix;
    int brushIndex;
    DebugInstanceData(const glm::mat4 &matrix, float sdf[8], int brushIndex);
};

// Inline simple implementations to keep declarations lightweight but linkable

inline OctreeNodeData::OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context, float * sdf) : level(level), node(node), cube(cube), containmentType(containmentType), context(context) {
    if (sdf) memcpy(this->sdf, sdf, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}

inline OctreeNodeData::OctreeNodeData(const OctreeNodeData &data) : level(data.level), node(data.node), cube(data.cube), containmentType(data.containmentType), context(data.context) {
    memcpy(this->sdf, data.sdf, sizeof(this->sdf));
}

inline OctreeNodeData::OctreeNodeData() : level(0), node(NULL), cube(), containmentType(ContainmentType::Intersects), context(NULL) {
    for(int i=0;i<8;++i) sdf[i] = INFINITY;
}

// OctreeNodeTriangleHandler ctor defined in OctreeNodeTriangleHandler.cpp

inline OctreeNodeFrame::OctreeNodeFrame() : node(NULL), cube(), level(0), brushIndex(DISCARD_BRUSH_INDEX), interpolated(false), chunkCube() {
    for(int i=0;i<8;++i) sdf[i] = INFINITY;
}

inline OctreeNodeFrame::OctreeNodeFrame(const OctreeNodeFrame &t) : node(t.node), cube(t.cube), level(t.level), brushIndex(t.brushIndex), interpolated(t.interpolated), chunkCube(t.chunkCube) {
    memcpy(this->sdf, t.sdf, sizeof(this->sdf));
}

inline OctreeNodeFrame::OctreeNodeFrame(OctreeNode* node, BoundingCube cube, uint level, float * sdf, int brushIndex, bool interpolated, BoundingCube chunkCube) : node(node), cube(cube), level(level), brushIndex(brushIndex), interpolated(interpolated), chunkCube(chunkCube) {
    if (sdf) memcpy(this->sdf, sdf, sizeof(this->sdf)); else for(int i=0;i<8;++i) this->sdf[i]=INFINITY;
}

inline NodeOperationResult::NodeOperationResult() : node(NULL), shapeType(SpaceType::Empty), resultType(SpaceType::Empty), process(false), isSimplified(false), brushIndex(DISCARD_BRUSH_INDEX) {
    for(int i=0;i<8;++i){ resultSDF[i]=INFINITY; shapeSDF[i]=INFINITY; }
}

inline NodeOperationResult::NodeOperationResult(OctreeNode * node, SpaceType shapeType, SpaceType resultType, const float * resultSDF, const float * shapeSDF, bool process, bool isSimplified, int brushIndex) : node(node), shapeType(shapeType), resultType(resultType), process(process), isSimplified(isSimplified), brushIndex(brushIndex) {
    if (resultSDF) memcpy(this->resultSDF, resultSDF, sizeof(this->resultSDF)); else for(int i=0;i<8;++i) this->resultSDF[i]=INFINITY;
    if (shapeSDF) memcpy(this->shapeSDF, shapeSDF, sizeof(this->shapeSDF)); else for(int i=0;i<8;++i) this->shapeSDF[i]=INFINITY;
}

inline ShapeArgs::ShapeArgs(float (*operation)(float, float), WrappedSignedDistanceFunction * function, const TexturePainter &painter, const Transformation model, glm::vec4 translate, glm::vec4 scale, Simplifier &simplifier, OctreeChangeHandler * changeHandler, float minSize) : operation(operation), function(function), painter(painter), model(model), translate(translate), scale(scale), simplifier(simplifier), changeHandler(changeHandler), minSize(minSize) {}

inline ThreadContext::ThreadContext(BoundingCube cube) : cube(cube) {}

inline InstanceData::InstanceData() : matrix(1.0f), shift(0.0f), animation(0) {}

inline InstanceData::InstanceData(uint animation, const glm::mat4 &matrix, float shift) : matrix(matrix), shift(shift), animation(animation) {}

inline DebugInstanceData::DebugInstanceData(const glm::mat4 &matrix, float sdf[8], int brushIndex) : matrix(matrix), brushIndex(brushIndex) {
    sdf1 = glm::vec4(sdf[0], sdf[1], sdf[2], sdf[3]);
    sdf2 = glm::vec4(sdf[4], sdf[5], sdf[6], sdf[7]);
}

inline OctreeNodeCubeSerialized::OctreeNodeCubeSerialized() : position(0.0f), normal(0.0f), texCoord(0.0f), brushIndex(DISCARD_BRUSH_INDEX), min(0.0f), bits(0), length(0.0f), level(0) {
    for(int i=0;i<8;++i) children[i]=UINT_MAX;
}

inline OctreeNodeCubeSerialized::OctreeNodeCubeSerialized(float * sdf, BoundingCube cube, Vertex vertex, uint bits, uint level) : position(vertex.position), normal(vertex.normal), texCoord(vertex.texCoord), brushIndex(vertex.brushIndex), min(cube.getCenter()), bits(bits), length(cube.getLength()), level(level) {
    for(int i=0;i<8;++i) children[i] = UINT_MAX;
}

#endif
