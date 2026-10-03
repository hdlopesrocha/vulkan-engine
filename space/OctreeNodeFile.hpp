#pragma once
#include "Octree.hpp"
#include "OctreeNodeSerialized.hpp"

class OctreeFile;

class OctreeNodeFile {
    OctreeNode * node;
    std::string filename;
    Octree * tree;

    // Recursive node-tree (de)serialization shared with OctreeFile. OctreeNodeFile
    // owns the recursion and OctreeFile delegates to it for its streamed trees.
    // `chunked` selects OctreeFile's chunk-file behaviour; when false the whole
    // subtree is walked inline (what a standalone node file needs).
    static OctreeNode * loadSubtree(Octree * tree, OctreeNode * workingNode, int i,
                                    const BoundingCube &cube,
                                    std::vector<OctreeNodeSerialized> * nodes,
                                    bool chunked, float chunkSize,
                                    const std::string &filename,
                                    const std::string &baseFolder);
    static uint saveSubtree(Octree * tree, OctreeNode * node,
                            std::vector<OctreeNodeSerialized> * nodes,
                            bool chunked, float chunkSize,
                            const std::string &filename,
                            const BoundingCube &cube,
                            const std::string &baseFolder);

    friend class OctreeFile;

public:
    OctreeNodeFile(Octree * tree, OctreeNode * node, std::string filename);
    void save(std::string baseFolder);
    void load(std::string baseFolder, const BoundingCube &cube);
};
