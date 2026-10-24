#pragma once

#include <vector>

#include "../space/Octree.hpp"

class SceneLoaderCallback {
public:
    SceneLoaderCallback() = default;
    ~SceneLoaderCallback() = default;

    // Dynamic layers: octrees[i] is scene layer i, with its update/delete
    // handlers at the same index. Implementations should tolerate any size
    // (>= 1): the default procedural loader populates layer 0 (opaque) and
    // layer 1 (transparent) when present and leaves extra layers empty.
    virtual void loadScene(
        std::vector<Octree*>& octrees,
        std::vector<Octree::OctreeNodeDataHandler>& updateHandlers,
        std::vector<Octree::OctreeNodeDataHandler>& deleteHandlers
    ) = 0;
};
