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
            
            // POM controls moved below and shown conditionally based on Mapping Mode
            ImGui::Spacing();
            ImGui::Text("Normal/Tangent Adjustments");
            ImGui::Separator();
            
            bool flipNormalYBool = (mat.flipNormalY > 0.5f);
            if (ImGui::Checkbox("Flip Normal Y", &flipNormalYBool)) {
                mat.flipNormalY = flipNormalYBool ? 1.0f : 0.0f;
            }
            
            bool flipTangentBool = (mat.flipTangentHandedness > 0.5f);
            if (ImGui::Checkbox("Flip Tangent Handedness", &flipTangentBool)) {
                mat.flipTangentHandedness = flipTangentBool ? 1.0f : 0.0f;
            }
            
            bool flipParallaxBool = (mat.flipParallaxDirection > 0.5f);
            if (ImGui::Checkbox("Flip Parallax Direction", &flipParallaxBool)) {
                mat.flipParallaxDirection = flipParallaxBool ? 1.0f : 0.0f;
            }
            
            ImGui::Spacing();
            ImGui::Text("Lighting");
            ImGui::Separator();
            
            ImGui::SliderFloat("Ambient Factor", &mat.ambientFactor, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Specular Strength", &mat.specularStrength, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Shininess", &mat.shininess, 1.0f, 256.0f, "%.0f");

            ImGui::Spacing();
            ImGui::Text("Mapping Mode");
            ImGui::Separator();
            // mappingMode: 0=none, 1=parallax, 2=tessellation
            int mappingMode = static_cast<int>(mat.mappingMode + 0.5f);
            const char* modes[] = { "None", "Parallax (POM)", "Tessellation (Displacement)" };
            if (ImGui::Combo("Mapping Mode", &mappingMode, modes, IM_ARRAYSIZE(modes))) {
                mat.mappingMode = static_cast<float>(mappingMode);
                // Default height interpretation: parallax uses legacy invert (black=deep),
                // tessellation uses direct (white=high). User can still override with the checkbox.
                mat.invertHeight = (mappingMode == 1) ? 0.0f : 1.0f;
            }
            // Height map interpretation: legacy inverted vs direct
            bool heightDirect = (mat.invertHeight > 0.5f);
            if (ImGui::Checkbox("Height Is Direct (white=high)", &heightDirect)) {
                mat.invertHeight = heightDirect ? 1.0f : 0.0f;
            }
            // Tessellation level (per-material). Only relevant when mappingMode == 2
            if (mappingMode == 2) {
                int tess = static_cast<int>(mat.tessLevel + 0.5f);
                if (ImGui::SliderInt("Tessellation Level", &tess, 1, 64)) {
                    mat.tessLevel = static_cast<float>(tess);
                }
                ImGui::SliderFloat("Tess Height Scale", &mat.tessHeightScale, 0.0f, 1.0f, "%.3f");
            }

            // Auto-set pomEnabled based on mapping mode and show POM sliders when appropriate
            mat.pomEnabled = (mappingMode == 1) ? 1.0f : 0.0f;
            if (mappingMode == 1) {
                ImGui::Separator();
                ImGui::Text("Parallax Occlusion Mapping");
                ImGui::SliderFloat("Height Scale", &mat.pomHeightScale, 0.0f, 0.2f, "%.3f");
                ImGui::SliderFloat("Min Layers", &mat.pomMinLayers, 1.0f, 64.0f, "%.0f");
                ImGui::SliderFloat("Max Layers", &mat.pomMaxLayers, 1.0f, 128.0f, "%.0f");
            }
            
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}
