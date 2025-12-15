#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include "Billboard.hpp"
#include "BillboardLayer.hpp"


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