#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "types/BillboardLayer.hpp"
#include "types/Billboard.hpp"

// Manages billboard instance positions for rendering and billboard layer definitions for the editor
class BillboardManager {
public:
    BillboardManager();

    // --- Authoring helpers used by the ImGui billboard creator ---
    size_t createBillboard(const std::string& name);
    void removeBillboard(size_t index);

    Billboard* getBillboard(size_t index);
    const Billboard* getBillboard(size_t index) const;
    size_t getBillboardCount() const;

    size_t addLayer(size_t billboardIndex, const BillboardLayer& layer);
    size_t addLayer(size_t billboardIndex, int atlasIndex, int tileIndex);
    void removeLayer(size_t billboardIndex, size_t layerIndex);

    BillboardLayer* getLayer(size_t billboardIndex, size_t layerIndex);
    const BillboardLayer* getLayer(size_t billboardIndex, size_t layerIndex) const;

    void moveLayerUp(size_t billboardIndex, size_t layerIndex);
    void moveLayerDown(size_t billboardIndex, size_t layerIndex);

    void clear();

private:
    // Editor billboard definitions
    std::vector<Billboard> billboards;
};
