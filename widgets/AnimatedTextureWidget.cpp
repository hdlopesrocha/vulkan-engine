#include "AnimatedTextureWidget.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <random>

AnimatedTextureWidget::AnimatedTextureWidget(std::shared_ptr<TextureMixer> textures_, const char* title)
    : Widget(title), textures(std::move(textures_)) {}

void AnimatedTextureWidget::render() {
    // Pass the widget's isOpen flag to ImGui so the window close button updates visibility
    if (!ImGui::Begin(title.c_str(), &isOpen, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("TextureTabBar")) {
        if (ImGui::BeginTabItem("Albedo")) {
            renderTextureTab(textures->getAlbedo(), 0);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            renderTextureTab(textures->getNormal(), 1);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Bump")) {
            renderTextureTab(textures->getBump(), 2);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }


    ImGui::Separator();

    ImGui::Text("Perlin Noise Generator");

    bool paramsChanged = false;

    ImGui::Separator();
    ImGui::Text("Noise Parameters");

    if (ImGui::SliderInt("Scale", &perlinScale, 1, 32)) paramsChanged = true;
    if (ImGui::SliderFloat("Octaves", &perlinOctaves, 1.0f, 8.0f)) paramsChanged = true;
    if (ImGui::SliderFloat("Persistence", &perlinPersistence, 0.0f, 1.0f)) paramsChanged = true;
    if (ImGui::SliderFloat("Lacunarity", &perlinLacunarity, 1.0f, 4.0f)) paramsChanged = true;
    if (ImGui::SliderFloat("Time", &perlinTime, 0.0f, 100.0f)) paramsChanged = true;

    ImGui::Separator();
    ImGui::Text("Adjustments");

    if (ImGui::SliderFloat("Brightness", &perlinBrightness, -1.0f, 1.0f)) paramsChanged = true;
    if (ImGui::SliderFloat("Contrast", &perlinContrast, 0.0f, 5.0f)) paramsChanged = true;

    MixerParameters params = MixerParameters({ 0, 1, 2, (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinSeed, perlinTime });

    if (paramsChanged) {
        // apply params to all textures
        textures->generatePerlinNoiseWithParams(1024, 1024, params);
    }

    ImGui::Separator();

    ImGui::End();
}

void AnimatedTextureWidget::renderTextureTab(EditableTexture& texture, int map) {
    float previewSize = 256.0f; // scaled down to 25%
    ImVec2 imageSize(previewSize, previewSize);
    VkDescriptorSet descSet = textures->getPreviewDescriptor(map);

    // Try normal descriptor
    if (descSet != VK_NULL_HANDLE) {
        ImGui::Image((ImTextureID)descSet, imageSize);
    } else {
        ImGui::Text("Texture preview not available");
    }

    ImGui::Text("Size: %dx%d", texture.getWidth(), texture.getHeight());

    const char* formatName = "Unknown";
    if (texture.getBytesPerPixel() == 4) {
        formatName = "RGBA8";
    } else if (texture.getBytesPerPixel() == 1) {
        formatName = "R8";
    }
    ImGui::Text("Format: %s", formatName);
}
