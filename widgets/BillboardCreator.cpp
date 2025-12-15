// Implementation for BillboardCreator - moved from header
#include "BillboardCreator.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <map>
#include <vector>
#include <cstdio>

// Constructor moved from header
BillboardCreator::BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureManager* textureMgr)
    : Widget("Billboard Creator"), billboardManager(billboardMgr), atlasManager(atlasMgr), textureManager(textureMgr) {
    isOpen = false; // Start closed
}

void BillboardCreator::setVulkanApp(VulkanApp* app) {
    vulkanApp = app;
    initializeTextures();
}

void BillboardCreator::initializeTextures() {
    if (!vulkanApp || texturesInitialized) return;

    const uint32_t texSize = 512; // Billboard texture size
    composedAlbedo.init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, "Billboard Albedo");
    composedNormal.init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, "Billboard Normal");
    composedOpacity.init(vulkanApp, texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, "Billboard Opacity");
    texturesInitialized = true;
}

void BillboardCreator::cleanup() {
    printf("[BillboardCreator] cleanup start: texturesInitialized=%d\n", texturesInitialized ? 1 : 0);
    if (texturesInitialized) {
        composedAlbedo.cleanup();
        composedNormal.cleanup();
        composedOpacity.cleanup();
        texturesInitialized = false;
    }
    printf("[BillboardCreator] cleanup done\n");
}

void BillboardCreator::renderBillboardEditor(Billboard* billboard) {
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
        selectedLayerIndex = billboardManager->addLayer(currentBillboardIndex, 0, 0);
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
                billboardManager->removeLayer(currentBillboardIndex, i);
                if (selectedLayerIndex >= static_cast<int>(billboard->layers.size())) {
                    selectedLayerIndex = billboard->layers.size() - 1;
                }
                ImGui::EndPopup();
                ImGui::PopID();
                break;
            }
            if (ImGui::MenuItem("Move Up", nullptr, false, i > 0)) {
                billboardManager->moveLayerUp(currentBillboardIndex, i);
            }
            if (ImGui::MenuItem("Move Down", nullptr, false, i < billboard->layers.size() - 1)) {
                billboardManager->moveLayerDown(currentBillboardIndex, i);
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
    ImGui::Text("Layer %d Properties", selectedLayerIndex);

    // Delete button
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button("Delete Layer")) {
        billboardManager->removeLayer(currentBillboardIndex, selectedLayerIndex);
        selectedLayerIndex = -1;
        ImGui::PopStyleColor(3);
        return;
    }
    ImGui::PopStyleColor(3);

    ImGui::Separator();

    // Atlas and tile selection
    ImGui::Text("Texture Selection:");
    const char* atlasNames[] = { "Foliage", "Grass", "Wild" };
    if (ImGui::Combo("Atlas", &layer->atlasIndex, atlasNames, 3)) {
        // Reset tile index when changing atlas
        layer->tileIndex = 0;
        needsRecomposition = true;
    }

    // Tile selection
    size_t tileCount = atlasManager->getTileCount(layer->atlasIndex);
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

    // Get the composed texture to display
    ImTextureID composedTexID = nullptr;
    if (previewTextureView == 0) {
        composedTexID = (ImTextureID)composedAlbedo.getImGuiDescriptorSet();
    } else if (previewTextureView == 1) {
        composedTexID = (ImTextureID)composedNormal.getImGuiDescriptorSet();
    } else {
        composedTexID = (ImTextureID)composedOpacity.getImGuiDescriptorSet();
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
    if (!texturesInitialized || !textureManager || !atlasManager) {
        printf("Cannot compose billboard: textures not initialized\n");
        return;
    }

    printf("=== Composing billboard: %s ===\n", billboard->name.c_str());
    printf("Billboard has %zu layers\n", billboard->layers.size());

    const uint32_t texSize = composedAlbedo.getWidth();
    printf("Output texture size: %u x %u\n", texSize, texSize);

    // Clear all textures to default values
    printf("Clearing textures to defaults...\n");
    clearTexture(composedAlbedo, 0, 0, 0, 0);           // Albedo: transparent
    clearTexture(composedNormal, 128, 128, 255, 255);   // Normal: up vector
    clearTexture(composedOpacity, 0, 0, 0, 255);        // Opacity: black

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
        compositeLayer(layer, tile, atlas, texSize);
    }

    // Free loaded texture data
    for (auto& [index, atlas] : atlasData) {
        freeAtlasTextures(atlas);
    }

    // Update GPU textures
    composedAlbedo.updateGPU();
    composedNormal.updateGPU();
    composedOpacity.updateGPU();

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
                                   const AtlasTextureData& atlas, uint32_t texSize) {
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
            sx /= layer->scaleX * tile->scaleX;
            sy /= layer->scaleY * tile->scaleY;

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
            blendPixel(composedAlbedo, dx, dy, srcR, srcG, srcB, finalOpacity);

            // Blend normal
            uint8_t normR = atlas.normalData[atlasIdx + 0];
            uint8_t normG = atlas.normalData[atlasIdx + 1];
            uint8_t normB = atlas.normalData[atlasIdx + 2];
            blendPixel(composedNormal, dx, dy, normR, normG, normB, finalOpacity);

            // Set opacity (max of current and new)
            uint8_t currentOp = composedOpacity.getPixelData()[dy * texSize * 4 + dx * 4];
            uint8_t newOp = (uint8_t)(finalOpacity * 255.0f);
            if (newOp > currentOp) {
                composedOpacity.setPixel(dx, dy, newOp, newOp, newOp, 255);
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
    if (!isOpen) return;
    if (!billboardManager || !atlasManager || !textureManager) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    // Left panel - Billboard list
    ImGui::BeginChild("BillboardList", ImVec2(200, 0), true);

    ImGui::Text("Billboards");
    ImGui::Separator();

    // Create new billboard button
    if (ImGui::Button("New Billboard", ImVec2(-1, 0))) {
        std::string name = "Billboard " + std::to_string(billboardManager->getBillboardCount() + 1);
        currentBillboardIndex = billboardManager->createBillboard(name);
        selectedLayerIndex = -1;
    }

    ImGui::Spacing();

    // List billboards
    for (size_t i = 0; i < billboardManager->getBillboardCount(); ++i) {
        const Billboard* billboard = billboardManager->getBillboard(i);
        if (!billboard) continue;

        ImGui::PushID(i);
        bool isSelected = (currentBillboardIndex == static_cast<int>(i));
        if (ImGui::Selectable(billboard->name.c_str(), isSelected)) {
            currentBillboardIndex = i;
            selectedLayerIndex = -1;
        }

        // Right-click menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Delete")) {
                billboardManager->removeBillboard(i);
                if (currentBillboardIndex >= static_cast<int>(billboardManager->getBillboardCount())) {
                    currentBillboardIndex = billboardManager->getBillboardCount() - 1;
                }
                selectedLayerIndex = -1;
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
            ImGui::End();
            return;
        }

        renderBillboardEditor(billboard);
    } else {
        ImGui::Text("Select or create a billboard to edit");
    }

    ImGui::EndChild();

    ImGui::End();
}
