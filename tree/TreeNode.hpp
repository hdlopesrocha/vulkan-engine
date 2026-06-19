#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace tree {

struct TreeNode {
    glm::vec3 position;
    float    radius;
    int      parent = -1;  // -1 = root
    bool     active = true; // can still spawn children

    TreeNode() : position(0.0f), radius(0.0f), parent(-1), active(true) {}
    TreeNode(const glm::vec3& pos, float r, int p, bool a = true)
        : position(pos), radius(r), parent(p), active(a) {}
};

} // namespace tree
