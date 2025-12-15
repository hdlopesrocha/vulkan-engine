#pragma once
#include "ChildBlock.hpp"
#include "../math/Vertex.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/SpaceType.hpp"

class OctreeAllocator;
class OctreeChangeHandler;
struct OctreeNodeCubeSerialized;

class OctreeNode {

public:
    Vertex vertex;
    uint id;
    uint8_t bits;
    float sdf[8];
    uint version;

    OctreeNode();
    OctreeNode(Vertex vertex);
    ~OctreeNode();
    OctreeNode * init(Vertex vertex);
    ChildBlock * clear(OctreeAllocator &allocator, OctreeChangeHandler * handler, ChildBlock * block);
    ChildBlock * getBlock(OctreeAllocator &allocator) const;
    ChildBlock * allocate(OctreeAllocator &allocator);
    void getChildren(OctreeAllocator &allocator, OctreeNode * childNodes[8]) const;
    void setChildren(OctreeAllocator &allocator, uint children[8]);

    void setType(SpaceType type);

    bool isSimplified() const;
    void setSimplified(bool value);

    bool isDirty() const ;
    void setDirty(bool value);

    bool isChunk() const ;
    void setChunk(bool value);

    bool isLeaf() const ;
    void setLeaf(bool value);

    SpaceType getType() const ;

    void setSDF(float value[8]);
    uint exportSerialization(OctreeAllocator &allocator, std::vector<OctreeNodeCubeSerialized> * nodes, int * leafNodes, BoundingCube cube, BoundingCube chunk, uint level);
    OctreeNode * compress(OctreeAllocator &allocator, BoundingCube * cube, BoundingCube chunk);
};

 
