#pragma once
#include "ChildBlock.hpp"
#include "../math/geometry/Vertex.hpp"
#include "../math/geometry/BoundingCube.hpp"
#include "../math/SpaceType.hpp"
#include <glm/glm.hpp>
#include <cstdint>

class OctreeAllocator;

class OctreeNode {

public:
    int brushIndex = 0;
    // Per-vertex texture-coordinate technique + UV painted by the TexturePainter
    // (TextureType values from math/TextureType.hpp). Copying them onto the node
    // lets the tessellators emit them as vertex attributes.
    int textureType = 0;
    glm::vec2 uv = glm::vec2(0.0f, 0.0f);
    glm::vec3 hsv = glm::vec3(0.0f, 0.5f, 0.5f);
    uint blockId;
    uint8_t bits;
    // Stored LoD levels are +1 shifted: getLod() returns the ladder level
    // incremented by one, so uint8_t squeezes levels 0..254 without a -1
    // sentinel — 0 declares "no LoD assigned" (unset/disable). Every
    // propagator (min(child)+1), walker comparison and store uses this
    // representation consistently.
    uint8_t lod = 0;
    uint8_t chunkLod = 0;
    float sdf[8];
    uint version;

    OctreeNode();
    OctreeNode(Vertex vertex);
    ~OctreeNode();
    OctreeNode * init(Vertex vert);
    ChildBlock * clear(OctreeAllocator &allocator, ChildBlock * block);
    ChildBlock * getBlock(OctreeAllocator &allocator) const;
    ChildBlock * allocate(OctreeAllocator &allocator);
    void getChildren(OctreeAllocator &allocator, OctreeNode * childNodes[8]) const;
    void setChildren(OctreeAllocator &allocator, OctreeNode * children[8]);

    void setType(SpaceType type);


    uint8_t getLod() const { return lod; }
    void setLod(uint8_t value) { lod = value; }

    uint8_t getChunkLod() const { return chunkLod; }
    void setChunkLod(uint8_t value) { chunkLod = value; }

    bool isChunk() const ;
    void setChunk(bool value);

    bool isLeaf() const ;

    void setBrush(int brushIndex);
    int getBrush() const;
    SpaceType getType() const ;

    void setSDF(float value[8]);
};

 
