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
    uint blockId;
    uint8_t bits;
    float sdf[8];
    uint version;

    OctreeNode();
    OctreeNode(Vertex vertex);
    ~OctreeNode();
    OctreeNode * init(Vertex vertex);
    ChildBlock * clear(OctreeAllocator &allocator, ChildBlock * block);
    ChildBlock * getBlock(OctreeAllocator &allocator) const;
    ChildBlock * allocate(OctreeAllocator &allocator);
    void getChildren(OctreeAllocator &allocator, OctreeNode * childNodes[8]) const;
    void setChildren(OctreeAllocator &allocator, uint children[8]);
    void setChildren(OctreeAllocator &allocator, OctreeNode * children[8]);

    void setType(SpaceType type);

    bool isSimplified() const;
    void setSimplified(bool value);

    bool isChunk() const ;
    void setChunk(bool value);

    bool isLeaf() const ;

    void setBrush(int brushIndex);
    int getBrush() const;
    SpaceType getType() const ;

    void setSDF(float value[8]);
    uint exportSerialization(OctreeAllocator &allocator, std::vector<OctreeNodeCubeSerialized> * nodes, int * leafNodes, const BoundingCube &cube, const BoundingCube &chunk, uint level);
    OctreeNode * compress(OctreeAllocator &allocator, BoundingCube * cube, const BoundingCube &chunk);
};

 
