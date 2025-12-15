#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

// Represents a single layer in a billboard
struct BillboardLayer {
    int atlasIndex = 0;        // Which atlas texture (0=foliage, 1=grass, 2=wild)
    int tileIndex = 0;         // Which tile from that atlas
    
    // Transform properties
    float offsetX = 0.0f;      // Position offset in billboard space (-1 to 1)
    float offsetY = 0.0f;
    float scaleX = 1.0f;       // Scale (1.0 = full size)
    float scaleY = 1.0f;
    float rotation = 0.0f;     // Rotation in degrees (0-360)
    
    // Rendering properties
    float opacity = 1.0f;      // Alpha multiplier (0-1)
    int renderOrder = 0;       // Higher values render on top
};

// Represents a complete billboard composed of multiple layers
struct Billboard {
    std::string name;
    std::vector<BillboardLayer> layers;
    
    // Billboard metadata
    float width = 1.0f;        // Physical width in world units
    float height = 1.0f;       // Physical height in world units
};

// Manages billboard definitions (no ImGui dependency)
class BillboardManager {
public:
    BillboardManager() = default;

    size_t createBillboard(const std::string& name);
    void removeBillboard(size_t index);

    Billboard* getBillboard(size_t index);
    const Billboard* getBillboard(size_t index) const;

    const std::vector<Billboard>& getBillboards() const;
    size_t getBillboardCount() const;

    size_t addLayer(size_t billboardIndex, const BillboardLayer& layer);
    size_t addLayer(size_t billboardIndex, int atlasIndex, int tileIndex);
    void removeLayer(size_t billboardIndex, size_t layerIndex);

    BillboardLayer* getLayer(size_t billboardIndex, size_t layerIndex);
    const BillboardLayer* getLayer(size_t billboardIndex, size_t layerIndex) const;

    void moveLayerUp(size_t billboardIndex, size_t layerIndex);
    void moveLayerDown(size_t billboardIndex, size_t layerIndex);

    void clear();

    std::string exportBillboard(size_t index) const;
    std::string exportAll() const;

    bool saveToFile(const std::string& filepath) const;
    bool loadFromFile(const std::string& filepath);

private:
    std::vector<Billboard> billboards;
};