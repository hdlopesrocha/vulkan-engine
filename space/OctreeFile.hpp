#pragma once
#include "Octree.hpp"
#include "OctreeNodeSerialized.hpp"

class OctreeFile {
    Octree * tree;
    std::string filename;
public:
    OctreeFile(Octree * tree, std::string filename);
    void writeToStream(std::ostream& out);
    void readFromStream(std::istream& in);
};
