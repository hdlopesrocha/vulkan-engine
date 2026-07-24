#pragma once

#include "Widget.hpp"
#include "../utils/BillboardManager.hpp"
#include "../utils/AtlasManager.hpp"
#include "../vulkan/TextureArrayManager.hpp"
#include "../vulkan/EditableTexture.hpp"
#include "../utils/AtlasTextureData.hpp"
#include "../services/BillboardService.hpp"
#include <imgui.h>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <array>
#include <memory>
// stb_image is included in the implementation file (BillboardCreator.cpp)

class BillboardCreator : public Widget {
public:
    BillboardCreator(BillboardManager* billboardMgr, AtlasManager* atlasMgr, TextureArrayManager* textureMgr,
                     std::shared_ptr<BillboardService> billboardService_);

    void setVulkanApp(class VulkanApp* app);
    void initializeTextures();
    void cleanup();
    void invalidateImGuiDescriptors();
    void bakeAllBillboards();
    const EditableTexture* getComposedAlbedoTexture(size_t index) const;

    VkImageView getAlbedoArrayView() const { return billboardService->getAlbedoArrayView(); }
    VkImageView getNormalArrayView()  const { return billboardService->getNormalArrayView(); }
    VkImageView getOpacityArrayView() const { return billboardService->getOpacityArrayView(); }
    VkSampler   getArraySampler()     const { return billboardService->getArraySampler(); }
    
    ~BillboardCreator() {
    }
    
    void render() override;
    
private:
    std::shared_ptr<BillboardService> billboardService;
    BillboardManager* billboardManager;
    AtlasManager* atlasManager;
    TextureArrayManager* textureManager;
    VulkanApp* vulkanApp = nullptr;
    
    int currentBillboardIndex = -1;
    int selectedLayerIndex = -1;
    int previewTextureView = 0;
    
    std::array<EditableTexture, 3> composedAlbedo;
    std::array<EditableTexture, 3> composedNormal;
    std::array<EditableTexture, 3> composedOpacity;
    bool texturesInitialized = false;
    bool needsRecomposition = true;
    
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
