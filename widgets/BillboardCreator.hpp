#pragma once

#include "Widget.hpp"
#include "../utils/BillboardManager.hpp"
#include "../vulkan/AtlasManager.hpp"
#include "../vulkan/TextureManager.hpp"
#include "../vulkan/EditableTexture.hpp"
#include "../utils/AtlasTextureData.hpp"
#include <imgui.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <stb_image.h>

class BillboardCreator : public Widget {
public:
    BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureManager* textureMgr);

    void setVulkanApp(class VulkanApp* app);
    void initializeTextures();
    void cleanup();
    
    ~BillboardCreator() {
        // Don't cleanup here - it will be done explicitly before VulkanApp destruction
    }
    
    void render() override;
    
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
    

    AtlasTextureData loadAtlasTextures(int atlasIndex);
    
    void freeAtlasTextures(AtlasTextureData& data);
    
    void clearTexture(EditableTexture& tex, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    
    void compositeLayer(const BillboardLayer* layer, const AtlasTile* tile, 
                       const AtlasTextureData& atlas, uint32_t texSize);
    
    void blendPixel(EditableTexture& tex, uint32_t x, uint32_t y, 
                   uint8_t srcR, uint8_t srcG, uint8_t srcB, float alpha);
    
    void drawCheckerboard(ImDrawList* drawList, ImVec2 pos, float size);
};
