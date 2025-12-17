#pragma once

#include "Widget.hpp"
#include "../vulkan/TextureArrayManager.hpp"
#include "../vulkan/AtlasManager.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <vector>
#include <string>
#include <vulkan/vulkan.h>

class VegetationAtlasEditor : public Widget {
public:
    VegetationAtlasEditor(TextureArrayManager* vegTextureArrayManager, AtlasManager* atlasManager);
    ~VegetationAtlasEditor();
    
    // We will request ImGui texture IDs from the TextureManager at render time.
    
    void render() override;
    
private:
    TextureArrayManager* vegetationTextureManager;
    AtlasManager* atlasManager;
    int currentTextureIndex = 0;  // Which vegetation texture (0=foliage, 1=grass, 2=wild)
    int selectedTileIndex = -1;   // Currently selected tile for editing
    int currentTextureView = 0;   // Which texture map to view (0=albedo, 1=normal, 2=opacity)
    
    // No manual descriptors: use TextureManager::getImTexture() at render time
};
