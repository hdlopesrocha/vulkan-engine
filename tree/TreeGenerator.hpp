#pragma once
#include "TreeNode.hpp"
#include "AttractorField.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <iostream>

namespace tree {

struct TreeParams {
    float influenceRadius = 64.0f;
    float killRadius      = 8.0f;
    float segmentLength   = 16.0f;
    int   maxIterations   = 200;
};

class TreeGenerator {
public:
    std::vector<TreeNode> nodes;
    AttractorField*       field = nullptr;
    TreeParams            params;

    void init(const glm::vec3& rootPos, float rootRadius) {
        nodes.clear();
        nodes.push_back(TreeNode(rootPos, rootRadius, -1, true));
    }

    void setField(AttractorField* f) { field = f; }

    int activeTipCount() const {
        int n = 0;
        for (auto& node : nodes) if (node.active) ++n;
        return n;
    }

    bool iterate() {
        if (!field || nodes.empty()) return false;

        std::vector<int> tips;
        for (size_t i = 0; i < nodes.size(); ++i)
            if (nodes[i].active) tips.push_back(static_cast<int>(i));
        if (tips.empty()) return false;

        // Assign each alive attractor to its nearest tip (within influence radius)
        std::vector<std::vector<int>> tipAttractors(tips.size());
        std::vector<bool> tipHasAny(tips.size(), false);

        for (size_t ai = 0; ai < field->points.size(); ++ai) {
            if (!field->points[ai].alive) continue;
            const glm::vec3& ap = field->points[ai].position;

            int   bestTip = -1;
            float bestDist = params.influenceRadius * params.influenceRadius;
            for (size_t ti = 0; ti < tips.size(); ++ti) {
                glm::vec3 d = ap - nodes[tips[ti]].position;
                float d2 = glm::dot(d, d);
                if (d2 < bestDist) { bestDist = d2; bestTip = static_cast<int>(ti); }
            }
            if (bestTip >= 0) {
                tipAttractors[bestTip].push_back(static_cast<int>(ai));
                tipHasAny[bestTip] = true;
            }
        }

        // Deactivate tips that got NO attractors — they've exhausted their region
        bool grew = false;
        for (size_t ti = 0; ti < tips.size(); ++ti) {
            if (!tipHasAny[ti]) {
                nodes[tips[ti]].active = false;
                continue;
            }
            if (tipAttractors[ti].empty()) continue;

            int tipIdx = tips[ti];

            // Average normalized direction to assigned attractors
            glm::vec3 dir(0.0f);
            for (int ai : tipAttractors[ti]) {
                glm::vec3 d = field->points[ai].position - nodes[tipIdx].position;
                float len = glm::length(d);
                if (len > 1e-6f) dir += d / len;
            }
            float dirLen = glm::length(dir);
            if (dirLen < 1e-6f) continue;
            dir /= dirLen;

            glm::vec3 newPos = nodes[tipIdx].position + dir * params.segmentLength;

            // Taper with depth
            int depth = 0;
            for (int p = nodes[tipIdx].parent; p >= 0; p = nodes[p].parent) ++depth;
            // Exponential taper: branches get exponentially thinner with depth
            float taper = std::pow(0.85f, static_cast<float>(depth));
            float newRadius = nodes[0].radius * taper;

            // Parent tip stays active for future branching; child is new active tip
            nodes.push_back(TreeNode(newPos, newRadius, tipIdx, true));
            grew = true;

            // Kill the attractors that were assigned to this tip.
            // Other tips' attractors survive → branching diversity.
            for (int ai : tipAttractors[ti]) {
                if (!field->points[ai].alive) continue;
                // Also require them to be within kill radius of the new node
                glm::vec3 d = newPos - field->points[ai].position;
                if (glm::dot(d, d) <= params.killRadius * params.killRadius)
                    field->points[ai].alive = false;
            }
        }
        return grew;
    }

    void grow() {
        std::cerr << "[TreeGenerator] root=(" << nodes[0].position.x
                  << "," << nodes[0].position.y << "," << nodes[0].position.z
                  << ") r=" << nodes[0].radius << " attractors=" << field->points.size() << std::endl;
        for (int iter = 0; iter < params.maxIterations; ++iter) {
            if (!iterate()) { std::cerr << "[TreeGenerator] iter " << iter << " no growth" << std::endl; break; }
            if (field && field->aliveCount() == 0) { std::cerr << "[TreeGenerator] iter " << iter << " no attractors" << std::endl; break; }
            if (activeTipCount() == 0) { std::cerr << "[TreeGenerator] iter " << iter << " no tips" << std::endl; break; }
        }
        std::cerr << "[TreeGenerator] done: " << nodes.size() << " nodes, "
                  << (field ? static_cast<int>(field->aliveCount()) : 0) << " alive"
                  << " r=" << nodes[0].radius << "→" << nodes.back().radius << std::endl;
    }
};

} // namespace tree
