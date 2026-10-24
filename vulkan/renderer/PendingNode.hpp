#pragma once

#include "../../space/OctreeNodeData.hpp"
#include "../../utils/Scene.hpp"

// One pending scene-graph change (layer + node data).
struct PendingNode {
    Layer layer;
    OctreeNodeData node;
};
