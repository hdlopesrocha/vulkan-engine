#include "AnimatedTextureWidget.hpp"
#include <imgui.h>
#include <random>

AnimatedTextureWidget::AnimatedTextureWidget(std::shared_ptr<EditableTextureSet> textures_, const char* title)
    : Widget(title), textures(std::move(textures_)) {}

void AnimatedTextureWidget::render() {
    if (!ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("TextureTabBar")) {
        if (ImGui::BeginTabItem("Albedo")) {
            renderTextureTab(textures->getAlbedo());
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            renderTextureTab(textures->getNormal());
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Bump")) {
            renderTextureTab(textures->getBump());
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void AnimatedTextureWidget::renderTextureTab(EditableTexture& texture) {
    ImGui::Text("Size: %dx%d", texture.getWidth(), texture.getHeight());

    const char* formatName = "Unknown";
    if (texture.getBytesPerPixel() == 4) {
        formatName = "RGBA8";
    } else if (texture.getBytesPerPixel() == 1) {
        formatName = "R8";
    }
    ImGui::Text("Format: %s", formatName);

    if (texture.getDirty()) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Texture has unsaved changes");
        if (ImGui::Button("Upload to GPU")) {
            texture.updateGPU();
        }
    } else {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Texture is up to date");
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

    if (paramsChanged) {
        // apply params to all textures
        textures->generatePerlinNoiseWithParams(textures->getAlbedo(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
        textures->generatePerlinNoiseWithParams(textures->getNormal(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
        textures->generatePerlinNoiseWithParams(textures->getBump(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
    }

    if (ImGui::Button("Generate Perlin Noise")) {
        textures->generatePerlinNoiseWithParams(texture, (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
    }
    ImGui::SameLine();
    if (ImGui::Button("Randomize Seed")) {
        std::random_device rd;
        perlinSeed = rd();
        textures->generatePerlinNoiseWithParams(texture, (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
    }
    ImGui::SameLine();
    if (ImGui::Button("Generate All")) {
        textures->generatePerlinNoiseWithParams(textures->getAlbedo(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
        textures->generatePerlinNoiseWithParams(textures->getNormal(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
        textures->generatePerlinNoiseWithParams(textures->getBump(), (float)perlinScale, perlinOctaves, perlinPersistence, perlinLacunarity, perlinBrightness, perlinContrast, perlinTime, perlinSeed);
    }

    ImGui::Separator();

    float previewSize = 512.0f;
    ImVec2 imageSize(previewSize, previewSize);
    ImGui::Text("Preview:");

    VkDescriptorSet descSet = texture.getImGuiDescriptorSet();
    if (descSet != VK_NULL_HANDLE) {
        ImGui::Image((ImTextureID)descSet, imageSize);
    } else {
        ImGui::Text("Texture preview not available");
    }
}
