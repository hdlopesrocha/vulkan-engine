#include "TextureViewerWidget.hpp"
#include <string>

void TextureViewer::render() {
    if (!manager) return;

    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }
    
    size_t tc = manager->count();
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
        if (ImGui::BeginTabItem("Albedo")) {
            ImTextureID tex = manager->getImTexture(currentIndex, 0);
            if (tex) ImGui::Image(tex, ImVec2(512,512));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            ImTextureID tex = manager->getImTexture(currentIndex, 1);
            if (tex) ImGui::Image(tex, ImVec2(512,512));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Height")) {
            ImTextureID tex = manager->getImTexture(currentIndex, 2);
            if (tex) ImGui::Image(tex, ImVec2(512,512));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Material")) {
            MaterialProperties& mat = manager->getMaterial(currentIndex);
            
            ImGui::Spacing();
            ImGui::Text("Normal/Tangent Adjustments");
            ImGui::Separator();
            

            bool triplanarBool = mat.triplanar;
            if (ImGui::Checkbox("Enable Triplanar Mapping", &triplanarBool)) {
                mat.triplanar = triplanarBool;
            }
            if (mat.triplanar) {
                ImGui::Indent();
                ImGui::SliderFloat("Triplanar Scale U", &mat.triplanarScaleU, 0.01f, 10.0f, "%.3f");
                ImGui::SliderFloat("Triplanar Scale V", &mat.triplanarScaleV, 0.01f, 10.0f, "%.3f");
                ImGui::Unindent();
            }
            
            ImGui::Spacing();
            ImGui::Text("Lighting");
            ImGui::Separator();
            
            ImGui::SliderFloat("Ambient Factor", &mat.ambientFactor, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Specular Strength", &mat.specularStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Shininess", &mat.shininess, 1.0f, 256.0f, "%.0f");

            ImGui::Spacing();
            ImGui::Text("Mapping (Tessellation + Bump)");
            ImGui::Separator();
            // mappingMode is now a boolean toggle
            bool mappingEnabled = mat.mappingMode;
            if (ImGui::Checkbox("Enable Tessellation & Bump Mapping", &mappingEnabled)) {
                mat.mappingMode = mappingEnabled;
            }
            // Height map interpretation: legacy inverted vs direct
            bool heightDirect = mat.invertHeight;
            if (ImGui::Checkbox("Height Is Direct (white=high)", &heightDirect)) {
                mat.invertHeight = heightDirect;
            }
            // Tessellation level (per-material). Only relevant when mapping is enabled
            if (mat.mappingMode) {
                int tess = static_cast<int>(mat.tessLevel + 0.5f);
                if (ImGui::SliderInt("Tessellation Level", &tess, 1, 64)) {
                    mat.tessLevel = static_cast<float>(tess);
                }
                ImGui::SliderFloat("Tess Height Scale", &mat.tessHeightScale, 0.0f, 1.0f, "%.3f");
            }
            
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}
