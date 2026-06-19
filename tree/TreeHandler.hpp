#pragma once
#include "TreeNode.hpp"
#include "AttractorField.hpp"
#include "TreeGenerator.hpp"
#include "../sdf/TaperedCapsuleDistanceFunction.hpp"
#include "../sdf/WrappedTaperedCapsule.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Transformation.hpp"
#include "../space/Simplifier.hpp"
#include "../space/Octree.hpp"
#include "../utils/SimpleBrush.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <iostream>

namespace tree {

class TreeHandler {
public:
    struct Segment {
        glm::vec3 start, end;
        float     r1, r2;  // radius at start / end
    };

    AttractorField field;
    TreeGenerator  generator;
    TreeParams     params;
    glm::vec3      rootPosition{0.0f};
    float          rootRadius = 16.0f;

    TreeHandler() {
        generator.setField(&field);
        generator.params = params;
    }

    void setParams(const TreeParams& p) {
        params = p;
        generator.params = params;
    }

    void setRoot(const glm::vec3& pos, float radius) {
        rootPosition = pos;
        rootRadius   = radius;
    }

    void populateSphere(const glm::vec3& center, float radius, int count, unsigned seed = 42) {
        field.populateSphere(center, radius, count, seed);
    }

    void populateBox(const glm::vec3& min, const glm::vec3& max, int count, unsigned seed = 42) {
        field.populateBox(min, max, count, seed);
    }

    void populateCrown(const glm::vec3& center, float radiusXZ, float radiusY,
                       int count, unsigned seed = 42) {
        field.populateCrown(center, radiusXZ, radiusY, count, seed);
    }

    void populateEllipsoid(const glm::vec3& center, float radiusXZ, float radiusY,
                           int count, unsigned seed = 42) {
        field.populateEllipsoid(center, radiusXZ, radiusY, count, seed);
    }

    void generate() {
        generator.init(rootPosition, rootRadius);
        generator.grow();
        std::cout << "[TreeHandler] Generated " << generator.nodes.size()
                  << " nodes, " << field.aliveCount() << " attractors remaining" << std::endl;
    }

    const std::vector<TreeNode>& getNodes() const { return generator.nodes; }

    // Extract all segments (parent → child) from the tree skeleton.
    std::vector<Segment> getSegments() const {
        std::vector<Segment> segs;
        for (size_t i = 1; i < generator.nodes.size(); ++i) {
            const auto& child  = generator.nodes[i];
            if (child.parent < 0) continue;
            const auto& parent = generator.nodes[child.parent];
            segs.push_back({
                parent.position, child.position,
                parent.radius,   child.radius
            });
        }
        return segs;
    }

    // Apply the tree skeleton as tapered capsules into a scene octree.
    // Nodes are in local space; the supplied transform places them in world space.
    void applyToOctree(
        Octree& octree,
        const OctreeChangeHandler& handler,
        const Transformation& model,
        const glm::vec4& translate,
        const glm::vec4& scale,
        int materialIndex,
        float minSize,
        Simplifier& simplifier
    ) {
        auto segs = getSegments();
        for (auto& seg : segs) {
            // Build SDF in local space (where the tree was generated)
            TaperedCapsuleDistanceFunction fn(seg.start, seg.end, seg.r1, seg.r2);
            WrappedTaperedCapsule wrapped(&fn);
            octree.apply(SDF::opUnion, &wrapped, model, translate, scale,
                         SimpleBrush(materialIndex), minSize, simplifier, handler);
        }
    }
};

} // namespace tree
