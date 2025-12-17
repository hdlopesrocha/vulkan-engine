#include "AnimatedTextureWidget.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <random>

AnimatedTextureWidget::AnimatedTextureWidget(std::shared_ptr<EditableTextureSet> textures_, const char* title)
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

    ImGui::End();
}

void AnimatedTextureWidget::renderTextureTab(EditableTexture& texture, int map) {
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

    // Debug info: prefer array-layer ImGui descriptor when available (keeps sRGB sampling consistent with cube rendering)
    VkDescriptorSet descSet = textures->getPreviewDescriptor(map);
    VkImageView view = texture.getView();
    VkSampler sampler = texture.getSampler();
    bool dirty = texture.getDirty();
    printf("[AnimatedTextureWidget] Preview info: desc=%p view=%p sampler=%p size=%dx%d bytes=%u dirty=%d map=%d\n",
           (void*)descSet, (void*)view, (void*)sampler, texture.getWidth(), texture.getHeight(), texture.getBytesPerPixel(), dirty ? 1 : 0, map);

    if (ImGui::Button("Refresh Preview")) {
        // Force upload and re-create ImGui descriptor if needed
        texture.updateGPU();
        VkDescriptorSet refreshed = texture.getImGuiDescriptorSet();
        printf("[AnimatedTextureWidget] Refresh requested, desc now=%p\n", (void*)refreshed);
    }
    ImGui::SameLine();
    if (ImGui::Button("Fill White")) {
        texture.fill(255,255,255,255);
        texture.updateGPU();
        printf("[AnimatedTextureWidget] Filled texture with white and uploaded to GPU\n");
    }
    ImGui::SameLine();
    if (ImGui::Button("Dump First Pixel")) {
        texture.debugDumpFirstPixel();
    }
    ImGui::SameLine();
    if (ImGui::Button("Render via texture.renderImGui()")) {
        // Fallback rendering path to validate ImGui call
        texture.renderImGui();
        printf("[AnimatedTextureWidget] Called texture.renderImGui()\n");
    }

    // Try normal descriptor
    if (descSet != VK_NULL_HANDLE) {
        ImGui::Image((ImTextureID)descSet, imageSize);
    } else {
        ImGui::Text("Texture preview not available");
    }

    // Debug: try creating a fresh ImGui descriptor from the current view/sampler and render it
    static VkDescriptorSet lastRecreated = VK_NULL_HANDLE;
    if (ImGui::Button("Recreate Descriptor")) {
        ImTextureID newId = ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        lastRecreated = (VkDescriptorSet)newId;
        printf("[AnimatedTextureWidget] Recreated ImGui descriptor %p and stored it\n", (void*)lastRecreated);
    }
    ImGui::SameLine();
    if (ImGui::Button("Render Recreated Descriptor")) {
        if (lastRecreated != VK_NULL_HANDLE) {
            ImGui::Image((ImTextureID)lastRecreated, imageSize);
            printf("[AnimatedTextureWidget] Rendered recreated descriptor %p\n", (void*)lastRecreated);
        } else {
            printf("[AnimatedTextureWidget] No recreated descriptor to render\n");
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Drop Recreated Descriptor")) {
        if (lastRecreated != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(lastRecreated);
            printf("[AnimatedTextureWidget] Removed recreated descriptor %p\n", (void*)lastRecreated);
            lastRecreated = VK_NULL_HANDLE;
        }
    }
}
