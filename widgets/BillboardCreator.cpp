// Implementation for BillboardCreator - moved from header
#include "BillboardCreator.hpp"
#include <cmath>
#include <cstring>
#include "../vulkan/VulkanApp.hpp"
#include <map>
#include <vector>
#include <cstdio>
#include <stb/stb_image.h>
#include "components/ImGuiHelpers.hpp"

// Constructor moved from header
BillboardCreator::BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureArrayManager* textureMgr)
    : Widget("Billboard Creator", u8"\uf03a"), billboardManager(billboardMgr), atlasManager(atlasMgr), textureManager(textureMgr) {
    isOpen = false;
}

void BillboardCreator::setVulkanApp(VulkanApp* app) {
    vulkanApp = app;
    initializeTextures();
}

void BillboardCreator::initializeTextures() {
    if (!vulkanApp || texturesInitialized) return;

    const uint32_t texSize = 512; // Billboard texture size
    for (size_t i = 0; i < composedAlbedo.size(); ++i) {
        const std::string suffix = " " + std::to_string(i);
        composedAlbedo[i].init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, ("Billboard Albedo" + suffix).c_str());
        composedNormal[i].init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, ("Billboard Normal" + suffix).c_str());
        composedOpacity[i].init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, ("Billboard Opacity" + suffix).c_str());
    }
    texturesInitialized = true;
}

void BillboardCreator::cleanup() {
    printf("[BillboardCreator] cleanup start: texturesInitialized=%d\n", texturesInitialized ? 1 : 0);
    if (texturesInitialized && vulkanApp) {
        VkDevice device = vulkanApp->getDevice();
        // Destroy array resources
        auto destroyAR = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
            if (view != VK_NULL_HANDLE) { vulkanApp->resources.removeImageView(view); vkDestroyImageView(device, view, nullptr); view = VK_NULL_HANDLE; }
            if (img  != VK_NULL_HANDLE) { vulkanApp->resources.removeImage(img);      vkDestroyImage(device, img, nullptr);          img  = VK_NULL_HANDLE; }
            if (mem  != VK_NULL_HANDLE) { vulkanApp->resources.removeDeviceMemory(mem); vkFreeMemory(device, mem, nullptr);           mem  = VK_NULL_HANDLE; }
        };
        destroyAR(billboardAlbedoArrayImage,  billboardAlbedoArrayMemory,  billboardAlbedoArrayView);
        destroyAR(billboardNormalArrayImage,   billboardNormalArrayMemory,  billboardNormalArrayView);
        destroyAR(billboardOpacityArrayImage,  billboardOpacityArrayMemory, billboardOpacityArrayView);
        if (billboardArraySampler != VK_NULL_HANDLE) {
            vulkanApp->resources.removeSampler(billboardArraySampler);
            vkDestroySampler(device, billboardArraySampler, nullptr);
            billboardArraySampler = VK_NULL_HANDLE;
        }
        for (size_t i = 0; i < composedAlbedo.size(); ++i) {
            composedAlbedo[i].cleanup();
            composedNormal[i].cleanup();
            composedOpacity[i].cleanup();
        }
        texturesInitialized = false;
    }
    printf("[BillboardCreator] cleanup done\n");
}

size_t BillboardCreator::getComposeIndex() const {
    if (currentBillboardIndex < 0) return 0;
    size_t idx = static_cast<size_t>(currentBillboardIndex);
    if (idx >= composedAlbedo.size()) idx = composedAlbedo.size() - 1;
    return idx;
}

void BillboardCreator::bakeAllBillboards() {
    if (!billboardManager || !vulkanApp) return;

    const size_t billboardCount = billboardManager->getBillboardCount();
    for (size_t i = 0; i < billboardCount; ++i) {
        if (i >= composedAlbedo.size()) {
            std::cerr << "[BillboardCreator] Skipping billboard " << i
                      << " (only " << composedAlbedo.size() << " direct textures are available)" << std::endl;
            break;
        }
        Billboard* billboard = billboardManager->getBillboard(i);
        if (!billboard) continue;
        currentBillboardIndex = static_cast<int>(i);
        composeBillboard(billboard);
    }

    createBillboardArrayTextures();
}

void BillboardCreator::createBillboardArrayTextures() {
    if (!vulkanApp || !texturesInitialized) return;

    VkDevice device = vulkanApp->getDevice();
    const uint32_t numLayers = static_cast<uint32_t>(composedAlbedo.size()); // 3
    const uint32_t w = composedAlbedo[0].getWidth();
    const uint32_t h = composedAlbedo[0].getHeight();
    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;

    // Destroy any previously created array resources
    auto destroyArrayResources = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (view != VK_NULL_HANDLE) {
            vulkanApp->resources.removeImageView(view);
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (img != VK_NULL_HANDLE) {
            vulkanApp->resources.removeImage(img);
            vkDestroyImage(device, img, nullptr);
            img = VK_NULL_HANDLE;
        }
        if (mem != VK_NULL_HANDLE) {
            vulkanApp->resources.removeDeviceMemory(mem);
            vkFreeMemory(device, mem, nullptr);
            mem = VK_NULL_HANDLE;
        }
    };
    destroyArrayResources(billboardAlbedoArrayImage, billboardAlbedoArrayMemory, billboardAlbedoArrayView);
    destroyArrayResources(billboardNormalArrayImage, billboardNormalArrayMemory, billboardNormalArrayView);
    destroyArrayResources(billboardOpacityArrayImage, billboardOpacityArrayMemory, billboardOpacityArrayView);
    if (billboardArraySampler != VK_NULL_HANDLE) {
        vulkanApp->resources.removeSampler(billboardArraySampler);
        vkDestroySampler(device, billboardArraySampler, nullptr);
        billboardArraySampler = VK_NULL_HANDLE;
    }

    // Helper: create a VkImage (2D array, no mipmaps) + VkImageView + allocate memory
    auto makeArrayImage = [&](VkImage& outImg, VkDeviceMemory& outMem, VkImageView& outView, const char* name) {
        VkImageCreateInfo imgInfo{};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = { w, h, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = numLayers;
        imgInfo.format = fmt;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        if (vkCreateImage(device, &imgInfo, nullptr, &outImg) != VK_SUCCESS)
            throw std::runtime_error(std::string("BillboardCreator: failed to create array image: ") + name);
        vulkanApp->resources.addImage(outImg, name);

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, outImg, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = vulkanApp->findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocInfo, nullptr, &outMem) != VK_SUCCESS)
            throw std::runtime_error(std::string("BillboardCreator: failed to allocate array memory: ") + name);
        vulkanApp->resources.addDeviceMemory(outMem, name);
        vkBindImageMemory(device, outImg, outMem, 0);

        // Transition all layers UNDEFINED -> SHADER_READ_ONLY_OPTIMAL (will be overwritten per-layer below)
        vulkanApp->transitionImageLayout(outImg, fmt, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, numLayers);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = outImg;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = fmt;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = numLayers;
        if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS)
            throw std::runtime_error(std::string("BillboardCreator: failed to create array image view: ") + name);
        vulkanApp->resources.addImageView(outView, name);
    };

    makeArrayImage(billboardAlbedoArrayImage,  billboardAlbedoArrayMemory,  billboardAlbedoArrayView,  "BillboardCreator: albedoArray");
    makeArrayImage(billboardNormalArrayImage,   billboardNormalArrayMemory,  billboardNormalArrayView,  "BillboardCreator: normalArray");
    makeArrayImage(billboardOpacityArrayImage,  billboardOpacityArrayMemory, billboardOpacityArrayView, "BillboardCreator: opacityArray");

    // Copy each billboard's composed texture into the corresponding array layer
    struct ChannelEntry { std::array<EditableTexture, 3>* src; VkImage dst; };
    ChannelEntry channels[3] = {
        { &composedAlbedo,  billboardAlbedoArrayImage  },
        { &composedNormal,  billboardNormalArrayImage  },
        { &composedOpacity, billboardOpacityArrayImage },
    };
    for (auto& ch : channels) {
        for (uint32_t layer = 0; layer < numLayers; ++layer) {
            EditableTexture& srcTex = (*ch.src)[layer];
            VkImage srcImage = srcTex.getImage();
            vulkanApp->runSingleTimeCommands([&](VkCommandBuffer cmd) {
                vulkanApp->recordTransitionImageLayoutLayer(cmd, srcImage, fmt,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 0, 1);
                VkImageCopy region{};
                region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.srcOffset = {0,0,0};
                region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1 };
                region.dstOffset = {0,0,0};
                region.extent = { w, h, 1 };
                vkCmdCopyImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               ch.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                vulkanApp->recordTransitionImageLayoutLayer(cmd, srcImage, fmt,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            });
        }
        // Transition entire array to SHADER_READ_ONLY_OPTIMAL
        vulkanApp->transitionImageLayout(ch.dst, fmt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, numLayers);
    }

    // Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &billboardArraySampler) != VK_SUCCESS)
        throw std::runtime_error("BillboardCreator: failed to create array sampler");
    vulkanApp->resources.addSampler(billboardArraySampler, "BillboardCreator: arraysampler");
    std::cerr << "[BillboardCreator] billboard array textures created: albedo=" << (void*)billboardAlbedoArrayView
              << " normal=" << (void*)billboardNormalArrayView
              << " opacity=" << (void*)billboardOpacityArrayView << std::endl;
}

const EditableTexture* BillboardCreator::getComposedAlbedoTexture(size_t index) const {
    if (index >= composedAlbedo.size()) return nullptr;
    return &composedAlbedo[index];
}

void BillboardCreator::renderBillboardEditor(Billboard* billboard) {
    if (!billboard || !billboardManager) return;

    // Billboard properties section
    ImGui::BeginChild("BillboardProperties", ImVec2(0, 120), true);

    ImGui::Text("Billboard: %s", billboard->name.c_str());
    ImGui::Separator();

    char nameBuffer[128];
    strncpy(nameBuffer, billboard->name.c_str(), sizeof(nameBuffer) - 1);
    nameBuffer[sizeof(nameBuffer) - 1] = '\0';
    if (ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer))) {
        billboard->name = nameBuffer;
    }

    ImGui::DragFloat("Width", &billboard->width, 0.1f, 0.1f, 10.0f, "%.2f");
    ImGui::DragFloat("Height", &billboard->height, 0.1f, 0.1f, 10.0f, "%.2f");

    ImGui::EndChild();

    ImGui::Spacing();

    // Preview section
    ImGui::BeginChild("BillboardPreview", ImVec2(0, 300), true);
    renderBillboardPreview(billboard);
    ImGui::EndChild();

    ImGui::Spacing();

    // Split into layers list and layer editor
    ImGui::BeginChild("LayersSection", ImVec2(250, 0), true);

    ImGui::Text("Layers (%zu)", billboard->layers.size());
    ImGui::Separator();

    // Add layer button
    if (ImGui::Button("Add Layer", ImVec2(-1, 0))) {
        size_t newLayerIndex = billboardManager->addLayer(static_cast<size_t>(currentBillboardIndex), 0, 0);
        if (newLayerIndex != static_cast<size_t>(-1)) {
            selectedLayerIndex = static_cast<int>(newLayerIndex);
        }
        needsRecomposition = true;
    }

    ImGui::Spacing();

    // List layers
    for (size_t i = 0; i < billboard->layers.size(); ++i) {
        ImGui::PushID(i);

        bool isSelected = (selectedLayerIndex == static_cast<int>(i));
        std::string layerLabel = "Layer " + std::to_string(i);

        if (ImGui::Selectable(layerLabel.c_str(), isSelected)) {
            selectedLayerIndex = i;
        }

        // Right-click menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Delete")) {
                billboardManager->removeLayer(static_cast<size_t>(currentBillboardIndex), i);
                if (billboard->layers.empty()) {
                    selectedLayerIndex = -1;
                } else if (selectedLayerIndex >= static_cast<int>(billboard->layers.size())) {
                    selectedLayerIndex = static_cast<int>(billboard->layers.size()) - 1;
                }
                needsRecomposition = true;
                ImGui::EndPopup();
                ImGui::PopID();
                break;
            }
            if (ImGui::MenuItem("Move Up", nullptr, false, i > 0)) {
                billboardManager->moveLayerUp(static_cast<size_t>(currentBillboardIndex), i);
                needsRecomposition = true;
            }
            if (ImGui::MenuItem("Move Down", nullptr, false, i < billboard->layers.size() - 1)) {
                billboardManager->moveLayerDown(static_cast<size_t>(currentBillboardIndex), i);
                needsRecomposition = true;
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    ImGui::EndChild();
    ImGui::SameLine();

    // Layer editor
    ImGui::BeginChild("LayerEditor", ImVec2(0, 0), true);

    if (selectedLayerIndex >= 0 && selectedLayerIndex < static_cast<int>(billboard->layers.size())) {
        BillboardLayer* layer = billboardManager->getLayer(currentBillboardIndex, selectedLayerIndex);
        if (layer) {
            renderLayerEditor(layer);
        }
    } else {
        ImGui::Text("Select or add a layer to edit");
    }

    ImGui::EndChild();
}

void BillboardCreator::renderLayerEditor(BillboardLayer* layer) {
    if (!layer || !billboardManager || !atlasManager) return;

    ImGui::Text("Layer %d Properties", selectedLayerIndex);

    // Delete button
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button("Delete Layer")) {
        billboardManager->removeLayer(static_cast<size_t>(currentBillboardIndex), static_cast<size_t>(selectedLayerIndex));
        selectedLayerIndex = -1;
        needsRecomposition = true;
        ImGui::PopStyleColor(3);
        return;
    }
    ImGui::PopStyleColor(3);

    ImGui::Separator();

    // Atlas and tile selection
    ImGui::Text("Texture Selection:");
    const char* atlasNames[] = { "Foliage", "Grass", "Wild" };
    if (layer->atlasIndex < 0 || layer->atlasIndex >= 3) {
        layer->atlasIndex = 0;
    }
    if (ImGui::Combo("Atlas", &layer->atlasIndex, atlasNames, 3)) {
        // Reset tile index when changing atlas
        layer->tileIndex = 0;
        needsRecomposition = true;
    }

    // Tile selection
    size_t tileCount = atlasManager->getTileCount(layer->atlasIndex);
    if (tileCount == 0) {
        layer->tileIndex = -1;
    } else if (layer->tileIndex < 0 || layer->tileIndex >= static_cast<int>(tileCount)) {
        layer->tileIndex = 0;
    }
    if (tileCount > 0) {
        ImGui::Text("Tile:");
        for (size_t i = 0; i < tileCount; ++i) {
            const AtlasTile* tile = atlasManager->getTile(layer->atlasIndex, i);
            if (!tile) continue;

            ImGui::PushID(i);
            if (ImGui::RadioButton(tile->name.c_str(), layer->tileIndex == static_cast<int>(i))) {
                layer->tileIndex = i;
                needsRecomposition = true;
            }
            ImGui::PopID();
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No tiles in this atlas");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Transform controls
    ImGui::Text("Transform:");
    needsRecomposition |= ImGui::DragFloat("Offset X", &layer->offsetX, 0.01f, -2.0f, 2.0f, "%.3f");
    needsRecomposition |= ImGui::DragFloat("Offset Y", &layer->offsetY, 0.01f, -2.0f, 2.0f, "%.3f");
    needsRecomposition |= ImGui::DragFloat("Scale X", &layer->scaleX, 0.01f, 0.01f, 5.0f, "%.3f");
    needsRecomposition |= ImGui::DragFloat("Scale Y", &layer->scaleY, 0.01f, 0.01f, 5.0f, "%.3f");
    needsRecomposition |= ImGui::SliderFloat("Rotation", &layer->rotation, 0.0f, 360.0f, "%.1f°");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Rendering properties
    ImGui::Text("Rendering:");
    needsRecomposition |= ImGui::SliderFloat("Opacity", &layer->opacity, 0.0f, 1.0f, "%.2f");
    needsRecomposition |= ImGui::InputInt("Render Order", &layer->renderOrder);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Quick transform presets
    ImGui::Text("Quick Presets:");
    if (ImGui::Button("Reset Transform")) {
        layer->offsetX = 0.0f;
        layer->offsetY = 0.0f;
        layer->scaleX = 1.0f;
        layer->scaleY = 1.0f;
        layer->rotation = 0.0f;
        needsRecomposition = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Flip H")) {
        layer->scaleX *= -1.0f;
        needsRecomposition = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Flip V")) {
        layer->scaleY *= -1.0f;
        needsRecomposition = true;
    }
}

void BillboardCreator::renderBillboardPreview(Billboard* billboard) {
    ImGui::Text("Billboard Preview (Composed)");
    ImGui::Separator();

    // Auto-compose if needed
    if (needsRecomposition && texturesInitialized && !billboard->layers.empty()) {
        composeBillboard(billboard);
    }

    // Texture view tabs
    if (ImGui::BeginTabBar("BillboardPreviewTabs")) {
        if (ImGui::BeginTabItem("Albedo")) {
            previewTextureView = 0;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Normal")) {
            previewTextureView = 1;
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Opacity")) {
            previewTextureView = 2;
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Spacing();

    // Show preview of composed billboard
    float previewSize = 256.0f;

    if (!texturesInitialized) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.0f, 1.0f), "Textures not initialized");
        ImGui::Text("Waiting for VulkanApp...");
        return;
    }

    if (billboard->layers.empty()) {
        ImVec2 cursorPos = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Draw background
        drawCheckerboard(drawList, cursorPos, previewSize);

        // Draw border
        drawList->AddRect(cursorPos, 
                         ImVec2(cursorPos.x + previewSize, cursorPos.y + previewSize),
                         IM_COL32(100, 100, 100, 255), 0.0f, 0, 2.0f);

        ImGui::Dummy(ImVec2(previewSize, previewSize));
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "No layers added yet");
        ImGui::Text("Add a layer to see the preview");
        return;
    }

    const size_t composeIndex = getComposeIndex();

    // Get the composed texture to display
    ImTextureID composedTexID = nullptr;
    if (previewTextureView == 0) {
        composedTexID = (ImTextureID)composedAlbedo[composeIndex].getImGuiDescriptorSet();
    } else if (previewTextureView == 1) {
        composedTexID = (ImTextureID)composedNormal[composeIndex].getImGuiDescriptorSet();
    } else {
        composedTexID = (ImTextureID)composedOpacity[composeIndex].getImGuiDescriptorSet();
    }

    if (composedTexID) {
        ImGui::Image(composedTexID, ImVec2(previewSize, previewSize));
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.0f, 1.0f), "Preview not available");
    }

    ImGui::Spacing();
    ImGui::Text("Layers: %zu", billboard->layers.size());
    ImGui::Text("View: %s", previewTextureView == 0 ? "Albedo" : 
                             previewTextureView == 1 ? "Normal" : "Opacity");

    // Recomposition status
    if (needsRecomposition) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Preview needs update!");
    } else {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Preview up to date");
    }

    // Button to recompose textures
    if (ImGui::Button("Recompose Billboard", ImVec2(-1, 0))) {
        composeBillboard(billboard);
    }
}

void BillboardCreator::composeBillboard(Billboard* billboard) {
    if (!billboard || !texturesInitialized || !atlasManager || !vulkanApp) {
        printf("Cannot compose billboard: textures not initialized\n");
        return;
    }

    printf("=== Composing billboard: %s ===\n", billboard->name.c_str());
    printf("Billboard has %zu layers\n", billboard->layers.size());

    const size_t composeIndex = getComposeIndex();
    EditableTexture& outAlbedo = composedAlbedo[composeIndex];
    EditableTexture& outNormal = composedNormal[composeIndex];
    EditableTexture& outOpacity = composedOpacity[composeIndex];

    const uint32_t texSize = outAlbedo.getWidth();
    if (texSize == 0) {
        printf("Cannot compose billboard: invalid texture size\n");
        return;
    }
    printf("Output texture size: %u x %u\n", texSize, texSize);

    // Clear all textures to default values
    printf("Clearing textures to defaults...\n");
    clearTexture(outAlbedo, 0, 0, 0, 0);           // Albedo: transparent
    clearTexture(outNormal, 128, 128, 255, 255);   // Normal: up vector
    clearTexture(outOpacity, 0, 0, 0, 255);        // Opacity: black

    // Sort layers by render order
    std::vector<std::pair<int, const BillboardLayer*>> sortedLayers;
    for (size_t i = 0; i < billboard->layers.size(); ++i) {
        sortedLayers.push_back({billboard->layers[i].renderOrder, &billboard->layers[i]});
    }
    std::sort(sortedLayers.begin(), sortedLayers.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Load atlas texture data
    printf("Loading atlas textures...\n");
    std::map<int, AtlasTextureData> atlasData;
    for (const auto& [order, layer] : sortedLayers) {
        if (layer->atlasIndex < 0 || layer->atlasIndex > 2) {
            continue;
        }
        if (layer->tileIndex < 0 || atlasManager->getTile(layer->atlasIndex, static_cast<size_t>(layer->tileIndex)) == nullptr) {
            continue;
        }
        if (atlasData.find(layer->atlasIndex) == atlasData.end()) {
            printf("Loading atlas %d...\n", layer->atlasIndex);
            atlasData[layer->atlasIndex] = loadAtlasTextures(layer->atlasIndex);
            if (atlasData[layer->atlasIndex].albedoData) {
                printf("  Atlas %d loaded: %dx%d\n", layer->atlasIndex, 
                       atlasData[layer->atlasIndex].width, atlasData[layer->atlasIndex].height);
            } else {
                printf("  WARNING: Failed to load atlas %d\n", layer->atlasIndex);
            }
        }
    }

    // Composite each layer
    printf("Compositing %zu layers...\n", sortedLayers.size());
    for (size_t i = 0; i < sortedLayers.size(); ++i) {
        const auto& [order, layer] = sortedLayers[i];
        printf("  Layer %zu: atlas=%d tile=%d order=%d\n", i, layer->atlasIndex, layer->tileIndex, order);

        const AtlasTile* tile = atlasManager->getTile(layer->atlasIndex, layer->tileIndex);
        if (!tile) {
            printf("    WARNING: Tile not found\n");
            continue;
        }

        const AtlasTextureData& atlas = atlasData[layer->atlasIndex];
        if (!atlas.albedoData || !atlas.normalData || !atlas.opacityData) {
            printf("    WARNING: Atlas data missing\n");
            continue;
        }

        printf("    Compositing tile (offset: %.2f,%.2f scale: %.2f,%.2f)...\n",
               tile->offsetX, tile->offsetY, tile->scaleX, tile->scaleY);
        compositeLayer(layer, tile, atlas, texSize, outAlbedo, outNormal, outOpacity);
    }

    // Free loaded texture data
    for (auto& [index, atlas] : atlasData) {
        freeAtlasTextures(atlas);
    }

    // Update GPU textures
    outAlbedo.updateGPU(vulkanApp);
    outNormal.updateGPU(vulkanApp);
    outOpacity.updateGPU(vulkanApp);

    needsRecomposition = false;
    printf("Billboard composition complete\n");
}

AtlasTextureData BillboardCreator::loadAtlasTextures(int atlasIndex) {
    AtlasTextureData data;

    std::string basePath = "textures/vegetation/";
    std::vector<std::string> atlasNames = {"foliage", "grass", "wild"};

    if (atlasIndex < 0 || atlasIndex >= (int)atlasNames.size()) {
        printf("  ERROR: Invalid atlas index %d\n", atlasIndex);
        return data;
    }

    std::string colorPath = basePath + atlasNames[atlasIndex] + "_color.jpg";
    std::string normalPath = basePath + atlasNames[atlasIndex] + "_normal.jpg";
    std::string opacityPath = basePath + atlasNames[atlasIndex] + "_opacity.jpg";

    printf("  Loading: %s\n", colorPath.c_str());
    printf("  Loading: %s\n", normalPath.c_str());
    printf("  Loading: %s\n", opacityPath.c_str());

    int channels;
    data.albedoData = stbi_load(colorPath.c_str(), &data.width, &data.height, &channels, 4);
    data.normalData = stbi_load(normalPath.c_str(), &data.width, &data.height, &channels, 4);
    data.opacityData = stbi_load(opacityPath.c_str(), &data.width, &data.height, &channels, 4);

    if (!data.albedoData || !data.normalData || !data.opacityData) {
        printf("  ERROR: Failed to load atlas textures for index %d\n", atlasIndex);
        if (!data.albedoData) printf("    - Color texture failed\n");
        if (!data.normalData) printf("    - Normal texture failed\n");
        if (!data.opacityData) printf("    - Opacity texture failed\n");
        freeAtlasTextures(data);
    }

    // Convert albedo (color) atlas texture from sRGB to linear so subsequent compositing uses linear values
    if (data.albedoData) {
        convertSRGB8ToLinearInPlace(data.albedoData, static_cast<size_t>(data.width) * static_cast<size_t>(data.height));
    }

    return data;
}

void BillboardCreator::freeAtlasTextures(AtlasTextureData& data) {
    if (data.albedoData) stbi_image_free(data.albedoData);
    if (data.normalData) stbi_image_free(data.normalData);
    if (data.opacityData) stbi_image_free(data.opacityData);
    data.albedoData = data.normalData = data.opacityData = nullptr;
}

void BillboardCreator::clearTexture(EditableTexture& tex, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    const uint32_t size = tex.getWidth();
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            tex.setPixel(x, y, r, g, b, a);
        }
    }
}

void BillboardCreator::compositeLayer(const BillboardLayer* layer, const AtlasTile* tile,
                                   const AtlasTextureData& atlas, uint32_t texSize,
                                   EditableTexture& outAlbedo,
                                   EditableTexture& outNormal,
                                   EditableTexture& outOpacity) {
    if (!layer || !tile || !atlas.albedoData || !atlas.normalData || !atlas.opacityData) return;
    if (atlas.width <= 0 || atlas.height <= 0 || texSize == 0) return;

    // For each pixel in the output texture
    for (uint32_t dy = 0; dy < texSize; ++dy) {
        for (uint32_t dx = 0; dx < texSize; ++dx) {
            // Convert output pixel to normalized coordinates (-1 to 1)
            float nx = (dx / (float)texSize) * 2.0f - 1.0f;
            float ny = 1.0f - (dy / (float)texSize) * 2.0f;

            // Apply inverse transform to get source coordinates
            float sx = nx - layer->offsetX;
            float sy = ny - layer->offsetY;

            // Apply rotation
            if (std::abs(layer->rotation) > 0.01f) {
                float rad = -layer->rotation * 3.14159f / 180.0f;
                float cosR = std::cos(rad);
                float sinR = std::sin(rad);
                float rx = sx * cosR - sy * sinR;
                float ry = sx * sinR + sy * cosR;
                sx = rx;
                sy = ry;
            }

            // Apply scale
            const float denomX = layer->scaleX * tile->scaleX;
            const float denomY = layer->scaleY * tile->scaleY;
            if (std::abs(denomX) < 1e-6f || std::abs(denomY) < 1e-6f) {
                continue;
            }

            sx /= denomX;
            sy /= denomY;

            // Check if within tile bounds (-1 to 1)
            if (sx < -1.0f || sx > 1.0f || sy < -1.0f || sy > 1.0f) continue;

            // Convert to UV coordinates (0 to 1)
            float u = (sx + 1.0f) * 0.5f;
            float v = (sy + 1.0f) * 0.5f;

            // Apply tile offset
            u = tile->offsetX + u * tile->scaleX;
            v = tile->offsetY + v * tile->scaleY;

            // Sample from atlas texture
            int ax = (int)(u * atlas.width);
            int ay = (int)(v * atlas.height);

            if (ax < 0 || ax >= atlas.width || ay < 0 || ay >= atlas.height) continue;

            int atlasIdx = (ay * atlas.width + ax) * 4;

            // Get opacity value (use red channel)
            uint8_t opacity = atlas.opacityData[atlasIdx];

            // Skip if fully transparent
            if (opacity == 0) continue;

            // Apply layer opacity
            float finalOpacity = (opacity / 255.0f) * layer->opacity;

            // Blend albedo
            uint8_t srcR = atlas.albedoData[atlasIdx + 0];
            uint8_t srcG = atlas.albedoData[atlasIdx + 1];
            uint8_t srcB = atlas.albedoData[atlasIdx + 2];
            blendPixel(outAlbedo, dx, dy, srcR, srcG, srcB, finalOpacity);

            // Blend normal
            uint8_t normR = atlas.normalData[atlasIdx + 0];
            uint8_t normG = atlas.normalData[atlasIdx + 1];
            uint8_t normB = atlas.normalData[atlasIdx + 2];
            blendPixel(outNormal, dx, dy, normR, normG, normB, finalOpacity);

            // Set opacity (max of current and new)
            const uint8_t* opacityPixels = outOpacity.getPixelData();
            if (!opacityPixels) continue;
            uint8_t currentOp = opacityPixels[dy * texSize * 4 + dx * 4];
            uint8_t newOp = (uint8_t)(finalOpacity * 255.0f);
            if (newOp > currentOp) {
                outOpacity.setPixel(dx, dy, newOp, newOp, newOp, 255);
            }
        }
    }
}

void BillboardCreator::blendPixel(EditableTexture& tex, uint32_t x, uint32_t y, 
                               uint8_t srcR, uint8_t srcG, uint8_t srcB, float alpha) {
    const uint32_t texSize = tex.getWidth();
    const uint8_t* data = tex.getPixelData();
    int idx = (y * texSize + x) * 4;

    uint8_t dstR = data[idx + 0];
    uint8_t dstG = data[idx + 1];
    uint8_t dstB = data[idx + 2];

    uint8_t outR = (uint8_t)(srcR * alpha + dstR * (1.0f - alpha));
    uint8_t outG = (uint8_t)(srcG * alpha + dstG * (1.0f - alpha));
    uint8_t outB = (uint8_t)(srcB * alpha + dstB * (1.0f - alpha));

    tex.setPixel(x, y, outR, outG, outB, 255);
}

void BillboardCreator::drawCheckerboard(ImDrawList* drawList, ImVec2 pos, float size) {
    if (previewTextureView == 1) {
        // Normal map: solid default "up" normal (0.5, 0.5, 1.0) = RGB(128, 128, 255)
        drawList->AddRectFilled(pos, 
                               ImVec2(pos.x + size, pos.y + size),
                               IM_COL32(128, 128, 255, 255));
    } else if (previewTextureView == 2) {
        // Opacity map: solid black (fully transparent)
        drawList->AddRectFilled(pos, 
                               ImVec2(pos.x + size, pos.y + size),
                               IM_COL32(0, 0, 0, 255));
    } else {
        // Albedo: checkerboard pattern to visualize transparency
        const int squares = 16;
        float squareSize = size / squares;

        for (int y = 0; y < squares; ++y) {
            for (int x = 0; x < squares; ++x) {
                ImVec2 p0(pos.x + x * squareSize, pos.y + y * squareSize);
                ImVec2 p1(p0.x + squareSize, p0.y + squareSize);

                ImU32 color = ((x + y) % 2 == 0) ? IM_COL32(200, 200, 200, 255) 
                                                  : IM_COL32(150, 150, 150, 255);
                drawList->AddRectFilled(p0, p1, color);
            }
        }
    }
}

// BillboardCreator::render moved from header
void BillboardCreator::render() {
    if (!billboardManager || !atlasManager) return;

    const size_t billboardCount = billboardManager->getBillboardCount();
    if (billboardCount == 0) {
        currentBillboardIndex = -1;
        selectedLayerIndex = -1;
    } else {
        if (currentBillboardIndex < 0 || currentBillboardIndex >= static_cast<int>(billboardCount)) {
            currentBillboardIndex = 0;
        }
        if (selectedLayerIndex >= 0) {
            Billboard* selectedBillboard = billboardManager->getBillboard(static_cast<size_t>(currentBillboardIndex));
            if (!selectedBillboard || selectedLayerIndex >= static_cast<int>(selectedBillboard->layers.size())) {
                selectedLayerIndex = -1;
            }
        }
    }

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // Left panel - Billboard list
    ImGui::BeginChild("BillboardList", ImVec2(200, 0), true);

    ImGui::Text("Billboards");
    ImGui::Separator();

    // Create new billboard button
    if (ImGui::Button("New Billboard", ImVec2(-1, 0))) {
        std::string name = "Billboard " + std::to_string(billboardManager->getBillboardCount() + 1);
        currentBillboardIndex = static_cast<int>(billboardManager->createBillboard(name));
        selectedLayerIndex = -1;
        needsRecomposition = true;
    }

    ImGui::Spacing();

    // List billboards
    for (size_t i = 0; i < billboardManager->getBillboardCount(); ++i) {
        const Billboard* billboard = billboardManager->getBillboard(i);
        if (!billboard) continue;

        ImGui::PushID(i);
        bool isSelected = (currentBillboardIndex == static_cast<int>(i));
        if (ImGui::Selectable(billboard->name.c_str(), isSelected)) {
            if (currentBillboardIndex != static_cast<int>(i)) {
                currentBillboardIndex = static_cast<int>(i);
                selectedLayerIndex = -1;
                needsRecomposition = true;
            }
        }

        // Right-click menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Delete")) {
                billboardManager->removeBillboard(i);
                const size_t updatedCount = billboardManager->getBillboardCount();
                if (updatedCount == 0) {
                    currentBillboardIndex = -1;
                } else if (currentBillboardIndex >= static_cast<int>(updatedCount)) {
                    currentBillboardIndex = static_cast<int>(updatedCount) - 1;
                }
                selectedLayerIndex = -1;
                needsRecomposition = true;
                ImGui::EndPopup();
                ImGui::PopID();
                break;
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    ImGui::EndChild();
    ImGui::SameLine();

    // Right panel - Billboard editor
    ImGui::BeginChild("BillboardEditor", ImVec2(0, 0), true);

    if (currentBillboardIndex >= 0 && currentBillboardIndex < static_cast<int>(billboardManager->getBillboardCount())) {
        Billboard* billboard = billboardManager->getBillboard(currentBillboardIndex);
        if (!billboard) {
            ImGui::Text("Error: Invalid billboard");
            ImGui::EndChild();
            return;
        }

        renderBillboardEditor(billboard);
    } else {
        ImGui::Text("Select or create a billboard to edit");
    }

    ImGui::EndChild();
}
