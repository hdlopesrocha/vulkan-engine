#pragma once

#include "Widget.hpp"
#include "../utils/BillboardManager.hpp"
#include "../vulkan/AtlasManager.hpp"
#include "../vulkan/TextureManager.hpp"
#include "../vulkan/EditableTexture.hpp"
#include <imgui.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <stb_image.h>

class BillboardCreator : public Widget {
public:
    BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureManager* textureMgr)
        : Widget("Billboard Creator"),
          billboardManager(billboardMgr),
          atlasManager(atlasMgr),
          textureManager(textureMgr) {
        isOpen = false; // Start closed
    }
    
    void setVulkanApp(VulkanApp* app) {
        vulkanApp = app;
        initializeTextures();
    }
    void initializeTextures();
    void cleanup();
    
    ~BillboardCreator() {
        // Don't cleanup here - it will be done explicitly before VulkanApp destruction
    }
    
    void render() override {
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
    
    // Declarations moved to .cpp
    
private:
    BillboardManager* billboardManager;
    AtlasManager* atlasManager;
    TextureManager* textureManager;
    VulkanApp* vulkanApp = nullptr;
    
    int currentBillboardIndex = -1;
    int selectedLayerIndex = -1;
    int previewTextureView = 0; // 0=albedo, 1=normal, 2=opacity
    
    // Composed billboard textures
    EditableTexture composedAlbedo;
    EditableTexture composedNormal;
    EditableTexture composedOpacity;
    bool texturesInitialized = false;
    bool needsRecomposition = true;
    
    void renderBillboardEditor(Billboard* billboard);
    
    void renderLayerEditor(BillboardLayer* layer);
    
    void renderBillboardPreview(Billboard* billboard);
    
    void composeBillboard(Billboard* billboard);
    
    struct AtlasTextureData {
        unsigned char* albedoData = nullptr;
        unsigned char* normalData = nullptr;
        unsigned char* opacityData = nullptr;
        int width = 0;
        int height = 0;
    };
    
    AtlasTextureData loadAtlasTextures(int atlasIndex);
    
    void freeAtlasTextures(AtlasTextureData& data);
    
    void clearTexture(EditableTexture& tex, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    
    void compositeLayer(const BillboardLayer* layer, const AtlasTile* tile, 
                       const AtlasTextureData& atlas, uint32_t texSize);
    
    void blendPixel(EditableTexture& tex, uint32_t x, uint32_t y, 
                   uint8_t srcR, uint8_t srcG, uint8_t srcB, float alpha);
    
    void drawCheckerboard(ImDrawList* drawList, ImVec2 pos, float size);
};
