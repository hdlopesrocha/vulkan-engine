#pragma once
#include "Octree.hpp"
#include "OctreeNodeSerialized.hpp"

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

 
