#include "BillboardManager.hpp"
#include <algorithm>

BillboardManager::BillboardManager() {}

// --- Authoring helpers (restored from legacy manager) ---

size_t BillboardManager::createBillboard(const std::string& name) {
    Billboard billboard;
    billboard.name = name;
    billboard.width = 1.0f;
    billboard.height = 1.0f;
    billboards.push_back(billboard);
    return billboards.size() - 1;
}

void BillboardManager::removeBillboard(size_t index) {
    if (index < billboards.size()) {
        billboards.erase(billboards.begin() + index);
    }
}

Billboard* BillboardManager::getBillboard(size_t index) {
    if (index < billboards.size()) {
        return &billboards[index];
    }
    return nullptr;
}

const Billboard* BillboardManager::getBillboard(size_t index) const {
    if (index < billboards.size()) {
        return &billboards[index];
    }
    return nullptr;
}

size_t BillboardManager::getBillboardCount() const {
    return billboards.size();
}

size_t BillboardManager::addLayer(size_t billboardIndex, const BillboardLayer& layer) {
    if (billboardIndex >= billboards.size()) return static_cast<size_t>(-1);
    billboards[billboardIndex].layers.push_back(layer);
    return billboards[billboardIndex].layers.size() - 1;
}

size_t BillboardManager::addLayer(size_t billboardIndex, int atlasIndex, int tileIndex) {
    if (billboardIndex >= billboards.size()) return static_cast<size_t>(-1);

    BillboardLayer layer;
    layer.atlasIndex = atlasIndex;
    layer.tileIndex = tileIndex;
    layer.offsetX = 0.0f;
    layer.offsetY = 0.0f;
    layer.scaleX = 1.0f;
    layer.scaleY = 1.0f;
    layer.rotation = 0.0f;
    layer.opacity = 1.0f;
    layer.renderOrder = static_cast<int>(billboards[billboardIndex].layers.size());

    billboards[billboardIndex].layers.push_back(layer);
    return billboards[billboardIndex].layers.size() - 1;
}

void BillboardManager::removeLayer(size_t billboardIndex, size_t layerIndex) {
    if (billboardIndex >= billboards.size()) return;
    if (layerIndex >= billboards[billboardIndex].layers.size()) return;
    billboards[billboardIndex].layers.erase(billboards[billboardIndex].layers.begin() + layerIndex);
}

BillboardLayer* BillboardManager::getLayer(size_t billboardIndex, size_t layerIndex) {
    if (billboardIndex >= billboards.size()) return nullptr;
    if (layerIndex >= billboards[billboardIndex].layers.size()) return nullptr;
    return &billboards[billboardIndex].layers[layerIndex];
}

const BillboardLayer* BillboardManager::getLayer(size_t billboardIndex, size_t layerIndex) const {
    if (billboardIndex >= billboards.size()) return nullptr;
    if (layerIndex >= billboards[billboardIndex].layers.size()) return nullptr;
    return &billboards[billboardIndex].layers[layerIndex];
}

void BillboardManager::moveLayerUp(size_t billboardIndex, size_t layerIndex) {
    if (billboardIndex >= billboards.size()) return;
    auto& layers = billboards[billboardIndex].layers;
    if (layerIndex == 0 || layerIndex >= layers.size()) return;
    std::swap(layers[layerIndex], layers[layerIndex - 1]);
}

void BillboardManager::moveLayerDown(size_t billboardIndex, size_t layerIndex) {
    if (billboardIndex >= billboards.size()) return;
    auto& layers = billboards[billboardIndex].layers;
    if (layerIndex >= layers.size() - 1) return;
    std::swap(layers[layerIndex], layers[layerIndex + 1]);
}

void BillboardManager::clear() {
    billboards.clear();
}

