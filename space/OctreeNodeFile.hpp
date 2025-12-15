#ifndef SPACE_OCTREE_NODE_FILE_HPP
#define SPACE_OCTREE_NODE_FILE_HPP

#include "types.hpp"
#include "octree.hpp"

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

#endif
