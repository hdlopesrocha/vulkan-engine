#include "TextureViewer.hpp"
#include <string>

void TextureViewer::render() {
    if (!manager) return;

    ImGui::Begin("Textures");
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
        ImGui::EndTabBar();
    }

    ImGui::End();
}
