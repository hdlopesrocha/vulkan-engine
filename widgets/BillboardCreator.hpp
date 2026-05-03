#pragma once

#include "Widget.hpp"
#include "../utils/BillboardManager.hpp"
#include "../utils/AtlasManager.hpp"
#include "../vulkan/TextureArrayManager.hpp"
#include "../vulkan/EditableTexture.hpp"
#include "../utils/AtlasTextureData.hpp"
#include <imgui.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <array>
// stb_image is included in the implementation file (BillboardCreator.cpp)

class BillboardCreator : public Widget {
public:
    BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureArrayManager* textureMgr);

    void setVulkanApp(class VulkanApp* app);
    void initializeTextures();
    void cleanup();
    void bakeAllBillboards();
    const EditableTexture* getComposedAlbedoTexture(size_t index) const;

    // Array textures: one VkImage per channel (albedo/normal/opacity), arrayLayers = numBillboards.
    // Filled by bakeAllBillboards(). Used by VegetationRenderer as sampler2DArray.
    VkImageView getAlbedoArrayView() const { return billboardAlbedoArrayView; }
    VkImageView getNormalArrayView()  const { return billboardNormalArrayView; }
    VkImageView getOpacityArrayView() const { return billboardOpacityArrayView; }
    VkSampler   getArraySampler()     const { return billboardArraySampler; }
    
    ~BillboardCreator() {
        // Don't cleanup here - it will be done explicitly before VulkanApp destruction
    }
    
    void render() override;
    
    // Declarations moved to .cpp
    
private:
    BillboardManager* billboardManager;
    AtlasManager* atlasManager;
    TextureArrayManager* textureManager;
    VulkanApp* vulkanApp = nullptr;
    
    int currentBillboardIndex = -1;
    int selectedLayerIndex = -1;
    int previewTextureView = 0; // 0=albedo, 1=normal, 2=opacity
    
    // Composed billboard textures (per-billboard, used as source for array copy)
    std::array<EditableTexture, 3> composedAlbedo;
    std::array<EditableTexture, 3> composedNormal;
    std::array<EditableTexture, 3> composedOpacity;
    bool texturesInitialized = false;
    bool needsRecomposition = true;

    // Billboard array images (sampler2DArray, one layer per billboard)
    VkImage        billboardAlbedoArrayImage   = VK_NULL_HANDLE;
    VkDeviceMemory billboardAlbedoArrayMemory  = VK_NULL_HANDLE;
    VkImageView    billboardAlbedoArrayView    = VK_NULL_HANDLE;
    VkImage        billboardNormalArrayImage   = VK_NULL_HANDLE;
    VkDeviceMemory billboardNormalArrayMemory  = VK_NULL_HANDLE;
    VkImageView    billboardNormalArrayView    = VK_NULL_HANDLE;
    VkImage        billboardOpacityArrayImage  = VK_NULL_HANDLE;
    VkDeviceMemory billboardOpacityArrayMemory = VK_NULL_HANDLE;
    VkImageView    billboardOpacityArrayView   = VK_NULL_HANDLE;
    VkSampler      billboardArraySampler       = VK_NULL_HANDLE;

    void createBillboardArrayTextures();
    
    void renderBillboardEditor(Billboard* billboard);
    
    void renderLayerEditor(BillboardLayer* layer);
    
    void renderBillboardPreview(Billboard* billboard);
    
    void composeBillboard(Billboard* billboard);
    size_t getComposeIndex() const;
    

    AtlasTextureData loadAtlasTextures(int atlasIndex);
    
    void freeAtlasTextures(AtlasTextureData& data);
    
    void clearTexture(EditableTexture& tex, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    
    void compositeLayer(const BillboardLayer* layer, const AtlasTile* tile,
                       const AtlasTextureData& atlas, uint32_t texSize,
                       EditableTexture& outAlbedo,
                       EditableTexture& outNormal,
                       EditableTexture& outOpacity);
    
    void blendPixel(EditableTexture& tex, uint32_t x, uint32_t y, 
                   uint8_t srcR, uint8_t srcG, uint8_t srcB, float alpha);
    
    void drawCheckerboard(ImDrawList* drawList, ImVec2 pos, float size);
};
