#pragma once
#include "Octree.hpp"
#include "OctreeNodeSerialized.hpp"

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

 
