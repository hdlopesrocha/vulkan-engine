#include "OctreeExplorerWidget.hpp"
#include <array>
#include "components/ImGuiHelpers.hpp"
#include "../math/Camera.hpp"
#include "../math/Ray.hpp"
#include <glm/glm.hpp>
#include <limits>

OctreeExplorerWidget::OctreeExplorerWidget(LocalScene* scene_, Camera* camera_)
    : Widget("Octree Explorer", u8"\uf1ad"), scene(scene_), camera(camera_) {}

const char* OctreeExplorerWidget::spaceTypeToString(SpaceType t) {
    switch (t) {
        case SpaceType::Empty: return "Empty";
        case SpaceType::Surface: return "Surface";
        case SpaceType::Solid: return "Solid";
        default: return "Unknown";
    }
}

void OctreeExplorerWidget::render() {
    if (!scene) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // Clear expanded cubes before rendering tree
    expandedCubes.clear();

    ImGui::Text("Layer: %s", selectedLayer == 0 ? "Opaque" : "Transparent");
    ImGui::SameLine();
    if (ImGui::BeginCombo("##layer", selectedLayer == 0 ? "Opaque" : "Transparent")) {
        if (ImGui::Selectable("Opaque", selectedLayer == 0)) selectedLayer = 0;
        if (ImGui::Selectable("Transparent", selectedLayer == 1)) selectedLayer = 1;
        ImGui::EndCombo();
    }

    ImGui::SliderInt("Max Depth", &maxDepth, 1, 32);
    ImGui::Checkbox("Show chunks only", &chunksOnly);
    ImGui::Checkbox("Show Debug Cubes", &showDebugCubes);
    ImGuiHelpers::SetTooltipIfHovered("Render debug octree/node cubes produced by the engine and explorer");

    // Expand all nodes on demand
    if (ImGui::Button("Expand All")) {
        expandAll = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Persist Expand", &expandAllPersistent)) {
        // toggled persistent expand
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        expandAllPersistent = false;
        collapseAll = true;
    }
    ImGuiHelpers::SetTooltipIfHovered("Expand/collapse helpers: Expand All = one frame, Persist Expand = keep expanded, Collapse All = collapse now");

    if (!octreeReady.load(std::memory_order_acquire)) {
        ImGui::TextDisabled("Loading scene...");
        return;
    }
    const Octree& tree = (selectedLayer == 0) ? scene->getOpaqueOctree() : scene->transparentOctree;
    handleRayExpandShortcut(tree);
    renderTree(tree);
}

bool OctreeExplorerWidget::buildMouseRay(const Octree& tree, Ray& outRay) const {
    if (!camera) return false;

    const ImGuiIO& io = ImGui::GetIO();
    ImGuiViewport* vp = ImGui::GetMainViewport();
    if (!vp) return false;
    if (vp->Size.x <= 0.0f || vp->Size.y <= 0.0f) return false;

    const float mouseX = io.MousePos.x - vp->Pos.x;
    const float mouseY = io.MousePos.y - vp->Pos.y;
    if (mouseX < 0.0f || mouseY < 0.0f || mouseX > vp->Size.x || mouseY > vp->Size.y) return false;

    // proj[1][1] *= -1 is baked in. Vulkan NDC: Y=-1 at top, Y=+1 at bottom.
    // Screen Y=0 (top)  → NDC Y = 2*0/h   - 1 = -1  ✓
    // Screen Y=h (bottom)→ NDC Y = 2*h/h   - 1 = +1  ✓
    // GLM_FORCE_DEPTH_ZERO_TO_ONE: far plane at NDC Z = 1.
    const float ndcX = (2.0f * mouseX) / vp->Size.x - 1.0f;
    const float ndcY = (2.0f * mouseY) / vp->Size.y - 1.0f;

    const glm::mat4 invVP = glm::inverse(camera->getViewProjectionMatrix());

    glm::vec4 farWorld = invVP * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    if (std::abs(farWorld.w) < 1e-8f) return false;
    farWorld /= farWorld.w;

    const glm::vec3 origin = camera->getPosition();
    const glm::vec3 dir = glm::vec3(farWorld) - origin;
    const float len2 = glm::dot(dir, dir);
    if (len2 <= std::numeric_limits<float>::epsilon()) return false;

    outRay = Ray(origin, glm::normalize(dir));
    return true;
}

bool OctreeExplorerWidget::updateRayOpenStateRecursive(OctreeNode* node, const BoundingCube& cube, OctreeAllocator* allocator, const Ray& ray) {
    if (!node) return false;

    if (!ray.intersects(cube)) {
        rayOpenState[node] = false;
        return false;
    }

    OctreeNode* children[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    node->getChildren(*allocator, children);

    bool hasAnyChild = false;
    for (int i = 0; i < 8; ++i) {
        OctreeNode* child = children[i];
        if (!child) continue;
        hasAnyChild = true;
        updateRayOpenStateRecursive(child, cube.getChild(i), allocator, ray);
    }

    if (hasAnyChild) {
        // Expand every internal node intersected by the ray.
        rayOpenState[node] = true;
    } else {
        // Mark intersecting leaves explicitly so filtering can be exact.
        rayOpenState[node] = true;
    }

    return true;
}

void OctreeExplorerWidget::handleRayExpandShortcut(const Octree& tree) {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    if (!ImGui::IsKeyPressed(ImGuiKey_Space, false)) return;

    if (!tree.root || !tree.allocator) return;

    Ray ray;
    if (!buildMouseRay(tree, ray)) return;

    rayOpenState.clear();
    rootRayOpen = updateRayOpenStateRecursive(tree.root, static_cast<const BoundingCube&>(tree), tree.allocator, ray);
    applyRayOpenState = true;
    expandAll = false;
    expandAllPersistent = false;
    collapseAll = false;
}

void OctreeExplorerWidget::renderTree(const Octree& tree) {
    OctreeNode* root = tree.root;
    if (!root) {
        ImGui::TextDisabled("Octree is empty");
        return;
    }

    OctreeAllocator* allocator = tree.allocator;
    BoundingCube rootCube = tree; // Octree inherits BoundingCube

    ImGui::Separator();
    ImGui::Text("Root: center=(%.1f, %.1f, %.1f) len=%.1f", rootCube.getCenter().x, rootCube.getCenter().y, rootCube.getCenter().z, rootCube.getLengthX());

    ImGuiTreeNodeFlags rootFlags = expandAll ? ImGuiTreeNodeFlags_DefaultOpen : 0;
    std::string label = "Level 0 | " + std::string(spaceTypeToString(root->getType()));

    // Honor expand/collapse before the root tree node (single-frame or persistent)
    if (applyRayOpenState) {
        ImGui::SetNextItemOpen(rootRayOpen, ImGuiCond_Always);
    } else if (collapseAll) {
        ImGui::SetNextItemOpen(false, ImGuiCond_Always);
    } else if (expandAll || expandAllPersistent) {
        ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    }

    bool open = ImGui::TreeNodeEx(label.c_str(), rootFlags);
    ImGui::SameLine();
    
    glm::vec3 nodeColor = getNodeColor(root);
    if (root->isChunk()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[chunk]"); ImGui::SameLine(); }
    if (root->isLeaf()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[leaf]"); ImGui::SameLine(); }
    if (root->isSimplified()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[simplified]"); ImGui::SameLine(); }
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "v%u", root->version);
    
    if (open) {
        renderNode(root, rootCube, 0, allocator, rootFlags);
        ImGui::TreePop();
    }

    // Reset single-frame expand/collapse flags
    if (expandAll) expandAll = false;
    if (collapseAll) collapseAll = false;
    if (applyRayOpenState) applyRayOpenState = false;
}

void OctreeExplorerWidget::renderNode(OctreeNode* node, const BoundingCube& cube, int depth, OctreeAllocator* allocator, ImGuiTreeNodeFlags extraFlags) {
    if (!node || depth >= maxDepth) return;

    OctreeNode* children[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    node->getChildren(*allocator, children);

    for (int i = 0; i < 8; ++i) {
        OctreeNode* child = children[i];
        if (!child) continue;

        BoundingCube childCube = cube.getChild(i);
        if (chunksOnly && !child->isChunk()) continue;

        bool hasGrandChildren = false;
        {
            OctreeNode* grand[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
            child->getChildren(*allocator, grand);
            for (auto* g : grand) {
                if (g) { hasGrandChildren = true; break; }
            }
        }

        ImGuiTreeNodeFlags flags = hasGrandChildren ? 0 : ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (expandAll) flags |= ImGuiTreeNodeFlags_DefaultOpen;
        std::string label = "Child " + std::to_string(i) + " | lvl=" + std::to_string(depth + 1)
            + " | " + spaceTypeToString(child->getType());

        glm::vec3 nodeColor = getNodeColor(child);

        if (flags & ImGuiTreeNodeFlags_Leaf) {
            ImGui::TreeNodeEx(label.c_str(), flags);
            ImGui::SameLine();
            if (child->isChunk()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[chunk]"); ImGui::SameLine(); }
            if (child->isLeaf()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[leaf]"); ImGui::SameLine(); }
            if (child->isSimplified()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[simplified]"); ImGui::SameLine(); }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "v%u", child->version);
        } else {
            // Apply expand/collapse to children (single-frame, persistent, or collapse)
            if (applyRayOpenState) {
                auto it = rayOpenState.find(child);
                const bool openByRay = (it != rayOpenState.end()) ? it->second : false;
                ImGui::SetNextItemOpen(openByRay, ImGuiCond_Always);
            } else if (collapseAll) {
                ImGui::SetNextItemOpen(false, ImGuiCond_Always);
            } else if (expandAll || expandAllPersistent) {
                ImGui::SetNextItemOpen(true, ImGuiCond_Always);
            }

            bool isExpanded = ImGui::TreeNodeEx(label.c_str(), flags);
            ImGui::SameLine();
            if (child->isChunk()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[chunk]"); ImGui::SameLine(); }
            if (child->isLeaf()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[leaf]"); ImGui::SameLine(); }
            if (child->isSimplified()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[simplified]"); ImGui::SameLine(); }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "v%u", child->version);
            
            if (isExpanded) {
                // Track this cube as expanded for debug visualization
                expandedCubes.push_back({childCube, getNodeColor(child)});
                renderNode(child, childCube, depth + 1, allocator, flags);
                ImGui::TreePop();
            }
        }
    }
}
