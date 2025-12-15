// Auto-generated wrapper header for OctreeChangeHandler
#pragma once

class OctreeNode;

class OctreeChangeHandler {
public:
    virtual void create(OctreeNode* nodeId) = 0;
    virtual void update(OctreeNode* nodeId) = 0;
    virtual void erase(OctreeNode* nodeId) = 0;
};