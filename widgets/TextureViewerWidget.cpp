#include "TextureViewerWidget.hpp"
#include <string>

void TextureViewer::render() {
    if (!arrayManager || !materials) return;

    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }
    
    size_t tc = materials->size();
    if (tc == 0) {
        ImGui::Text("No textures loaded");
        ImGui::End();
        return;
    }

    // clamp currentIndex
    if (currentIndex >= tc) currentIndex = 0;

    if (ImGui::Button("<")) {
        if (currentIndex == 0) currentIndex = tc - 1;
        else --currentIndex;
    }
    ImGui::SameLine();
    ImGui::Text("%zu / %zu", currentIndex + 1, tc);
    ImGui::SameLine();
    if (ImGui::Button(">")) {
        currentIndex = (currentIndex + 1) % tc;
    }

    std::string tabBarId = std::string("tabs_") + std::to_string(currentIndex);
    if (ImGui::BeginTabBar(tabBarId.c_str())) {
        const float previewSize = 256.0f; // 25% of previous size
        if (ImGui::BeginTabItem("Albedo")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 0);
            if (tex) ImGui::Image(tex, ImVec2(previewSize, previewSize));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 1);
            if (tex) ImGui::Image(tex, ImVec2(previewSize, previewSize));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Height")) {
            ImTextureID tex = arrayManager->getImTexture(currentIndex, 2);
            if (tex) ImGui::Image(tex, ImVec2(previewSize, previewSize));
            ImGui::EndTabItem();
        }
        // Material tab removed from tabs — material UI will be shown under the preview
        ImGui::EndTabBar();

        // Material controls shown under the preview tabs
        {
            MaterialProperties& mat = (*materials)[currentIndex];
            bool dirty = false;

            ImGui::Spacing();
            ImGui::Text("Material");
            ImGui::Separator();

            ImGui::Text("Normal/Tangent Adjustments");
            ImGui::Separator();

            bool triplanarBool = mat.triplanar;
            if (ImGui::Checkbox("Enable Triplanar Mapping", &triplanarBool)) { mat.triplanar = triplanarBool; dirty = true; }
            if (mat.triplanar) {
                ImGui::Indent();
                if (ImGui::SliderFloat("Triplanar Scale U", &mat.triplanarScaleU, 0.01f, 10.0f, "%.3f")) dirty = true;
                if (ImGui::SliderFloat("Triplanar Scale V", &mat.triplanarScaleV, 0.01f, 10.0f, "%.3f")) dirty = true;
                ImGui::Unindent();
            }

            ImGui::Spacing();
            ImGui::Text("Lighting");
            ImGui::Separator();

            if (ImGui::SliderFloat("Ambient Factor", &mat.ambientFactor, 0.0f, 1.0f, "%.2f")) dirty = true;
            if (ImGui::SliderFloat("Specular Strength", &mat.specularStrength, 0.0f, 2.0f, "%.2f")) dirty = true;
            if (ImGui::SliderFloat("Shininess", &mat.shininess, 1.0f, 256.0f, "%.0f")) dirty = true;

            ImGui::Spacing();
            ImGui::Text("Mapping (Tessellation + Bump)");
            ImGui::Separator();
            bool mappingEnabled = mat.mappingMode;
            if (ImGui::Checkbox("Enable Tessellation & Bump Mapping", &mappingEnabled)) { mat.mappingMode = mappingEnabled; dirty = true; }
            bool heightDirect = mat.invertHeight;
            if (ImGui::Checkbox("Height Is Direct (white=high)", &heightDirect)) { mat.invertHeight = heightDirect; dirty = true; }
            if (mat.mappingMode) {
                int tess = static_cast<int>(mat.tessLevel + 0.5f);
                if (ImGui::SliderInt("Tessellation Level", &tess, 1, 64)) { mat.tessLevel = static_cast<float>(tess); dirty = true; }
                if (ImGui::SliderFloat("Tess Height Scale", &mat.tessHeightScale, 0.0f, 1.0f, "%.3f")) dirty = true;
            }

            if (dirty && onMaterialChanged) onMaterialChanged(currentIndex);
        }
    }

    ImGui::End();
}

TextureViewer::TextureViewer() : Widget("Textures") {}

void TextureViewer::init(TextureArrayManager* arrayManager, std::vector<MaterialProperties>* materials) {
    this->arrayManager = arrayManager;
    this->materials = materials;
}
