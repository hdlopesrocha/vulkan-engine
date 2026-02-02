#include "OctreeExplorerWidget.hpp"
#include <array>

OctreeExplorerWidget::OctreeExplorerWidget(LocalScene* scene_)
    : Widget("Octree Explorer"), scene(scene_) {}

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
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

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

    const Octree& tree = (selectedLayer == 0) ? scene->getOpaqueOctree() : scene->transparentOctree;
    renderTree(tree);

    ImGui::End();
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
    bool open = ImGui::TreeNodeEx(label.c_str(), rootFlags);
    ImGui::SameLine();
    
    glm::vec3 nodeColor = getNodeColor(root);
    if (root->isChunk()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[chunk]"); ImGui::SameLine(); }
    if (root->isLeaf()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[leaf]"); ImGui::SameLine(); }
    if (root->isDirty()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[dirty]"); ImGui::SameLine(); }
    if (root->isSimplified()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[simplified]"); ImGui::SameLine(); }
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "v%u", root->version);
    
    if (open) {
        renderNode(root, rootCube, 0, allocator, rootFlags);
        ImGui::TreePop();
    }
    if (expandAll) expandAll = false;
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
            if (child->isDirty()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[dirty]"); ImGui::SameLine(); }
            if (child->isSimplified()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[simplified]"); ImGui::SameLine(); }
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "v%u", child->version);
        } else {
            bool isExpanded = ImGui::TreeNodeEx(label.c_str(), flags);
            ImGui::SameLine();
            if (child->isChunk()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[chunk]"); ImGui::SameLine(); }
            if (child->isLeaf()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[leaf]"); ImGui::SameLine(); }
            if (child->isDirty()) { ImGui::TextColored(ImVec4(nodeColor.r, nodeColor.g, nodeColor.b, 1.0f), "[dirty]"); ImGui::SameLine(); }
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
