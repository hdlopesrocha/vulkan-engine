#pragma once

#include "Widget.hpp"
#include "../utils/LocalScene.hpp"
#include <imgui.h>
#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>

class Camera;
class Ray;

// Simple ImGui widget to explore an octree recursively.
// Starts collapsed; shows node type/chunk flags and children.
class OctreeExplorerWidget : public Widget {
public:
    explicit OctreeExplorerWidget(LocalScene* scene, Camera* camera = nullptr);
    void render() override;
    void setCamera(Camera* camera_) { camera = camera_; }
    bool getShowDebugCubes() const { return showDebugCubes; }

    // Set to false while the octree is being written by a background thread.
    // The widget will skip accessing the octree until this becomes true.
    std::atomic<bool> octreeReady{false};

    struct CubeWithColor {
        BoundingCube cube;
        glm::vec3 color;
    };

    // Get list of cubes for expanded nodes (for debug visualization)
    const std::vector<CubeWithColor>& getExpandedCubes() const { return expandedCubes; }

private:
    LocalScene* scene; // not owned
    int selectedLayer = 0; // 0 = opaque, 1 = transparent
    int maxDepth = 10;     // safety cap for very deep trees
    bool chunksOnly = false;
    std::vector<CubeWithColor> expandedCubes; // Cubes of expanded nodes with colors
    bool expandAll = false; // Expand all nodes on first render
    bool expandAllPersistent = false; // Persistently keep nodes expanded
    bool collapseAll = false; // Trigger collapse of all nodes for one frame
    bool showDebugCubes = false; // Debug cubes visibility for octree explorer
    bool applyRayOpenState = false; // Apply ray-based open/closed state in next render pass
    bool rootRayOpen = false;
    std::unordered_map<const OctreeNode*, bool> rayOpenState;
    
    // Color priority (higher = more important): dirty > chunk > simplified > leaf
    glm::vec3 getNodeColor(OctreeNode* node) const {
        if (node->isChunk()) return glm::vec3(1.0f, 0.0f, 0.0f);        // Red
        if (node->isSimplified()) return glm::vec3(0.0f, 1.0f, 0.0f);   // Green
        if (node->isLeaf()) return glm::vec3(0.0f, 0.0f, 1.0f);         // Blue
        return glm::vec3(0.5f, 0.5f, 0.5f);                             // Gray - default
    }

    void renderTree(const Octree& tree);
    void renderNode(OctreeNode* node, const BoundingCube& cube, int depth, OctreeAllocator* allocator, ImGuiTreeNodeFlags extraFlags = 0);
    void handleRayExpandShortcut(const Octree& tree);
    bool buildMouseRay(const Octree& tree, Ray& outRay) const;
    bool updateRayOpenStateRecursive(OctreeNode* node, const BoundingCube& cube, OctreeAllocator* allocator, const Ray& ray);
    static const char* spaceTypeToString(SpaceType t);

    Camera* camera = nullptr; // not owned
};
