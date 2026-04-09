#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "BillboardBaseModel.hpp"
#include "../utils/BillboardLayer.hpp"
#include "../utils/Billboard.hpp"

// Manages billboard instance positions for rendering and billboard layer definitions for the editor
class BillboardManager {
public:
    BillboardManager();

    // --- Runtime billboard placement (used by vegetation/renderer) ---
    void setCandidatePositions(const std::vector<glm::vec3>& positions);
    void setBillboardCount(size_t count);
    const std::vector<glm::vec3>& getBillboardPositions() const;
    void updateBillboardSelection();

    // --- Authoring helpers used by the ImGui billboard creator ---
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
    // Runtime sampling data
    std::vector<glm::vec3> candidatePositions;
    std::vector<glm::vec3> billboardPositions;
    size_t billboardCount;

    // Editor billboard definitions
    std::vector<Billboard> billboards;
};
