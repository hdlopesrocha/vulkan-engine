#include "BillboardManager.hpp"
#include <sstream>
#include <fstream>

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

const std::vector<Billboard>& BillboardManager::getBillboards() const {
    return billboards;
}

size_t BillboardManager::getBillboardCount() const {
    return billboards.size();
}

size_t BillboardManager::addLayer(size_t billboardIndex, const BillboardLayer& layer) {
    if (billboardIndex >= billboards.size()) return (size_t)-1;
    billboards[billboardIndex].layers.push_back(layer);
    return billboards[billboardIndex].layers.size() - 1;
}

size_t BillboardManager::addLayer(size_t billboardIndex, int atlasIndex, int tileIndex) {
    if (billboardIndex >= billboards.size()) return (size_t)-1;

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

std::string BillboardManager::exportBillboard(size_t index) const {
    if (index >= billboards.size()) return "";

    const Billboard& billboard = billboards[index];
    std::stringstream ss;

    ss << "# Billboard: " << billboard.name << "\n";
    ss << "width=" << billboard.width << "\n";
    ss << "height=" << billboard.height << "\n";
    ss << "layers=" << billboard.layers.size() << "\n";

    for (size_t i = 0; i < billboard.layers.size(); ++i) {
        const auto& layer = billboard.layers[i];
        ss << "layer," << i << ","
           << layer.atlasIndex << ","
           << layer.tileIndex << ","
           << layer.offsetX << ","
           << layer.offsetY << ","
           << layer.scaleX << ","
           << layer.scaleY << ","
           << layer.rotation << ","
           << layer.opacity << ","
           << layer.renderOrder << "\n";
    }

    return ss.str();
}

std::string BillboardManager::exportAll() const {
    std::stringstream ss;
    ss << "# Billboard Collection\n";
    ss << "count=" << billboards.size() << "\n\n";

    for (size_t i = 0; i < billboards.size(); ++i) {
        ss << exportBillboard(i) << "\n";
    }

    return ss.str();
}

bool BillboardManager::saveToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;
    file << exportAll();
    file.close();
    return true;
}

bool BillboardManager::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    billboards.clear();
    std::string line;
    Billboard* currentBillboard = nullptr;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Parse key=value lines
        size_t equalPos = line.find('=');
        if (equalPos != std::string::npos) {
            std::string key = line.substr(0, equalPos);
            std::string value = line.substr(equalPos + 1);

            if (key == "width" && currentBillboard) {
                currentBillboard->width = std::stof(value);
            } else if (key == "height" && currentBillboard) {
                currentBillboard->height = std::stof(value);
            } else if (key == "layers") {
                // Start a new billboard
                billboards.push_back(Billboard());
                currentBillboard = &billboards.back();
                currentBillboard->name = "Billboard " + std::to_string(billboards.size());
            }
            continue;
        }

        // Parse layer lines
        if (line.substr(0, 6) == "layer," && currentBillboard) {
            std::stringstream ss(line.substr(6));
            std::string token;
            std::vector<std::string> tokens;

            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            if (tokens.size() >= 9) {
                BillboardLayer layer;
                // tokens[0] is index, skip it
                layer.atlasIndex = std::stoi(tokens[1]);
                layer.tileIndex = std::stoi(tokens[2]);
                layer.offsetX = std::stof(tokens[3]);
                layer.offsetY = std::stof(tokens[4]);
                layer.scaleX = std::stof(tokens[5]);
                layer.scaleY = std::stof(tokens[6]);
                layer.rotation = std::stof(tokens[7]);
                layer.opacity = std::stof(tokens[8]);
                if (tokens.size() >= 10) {
                    layer.renderOrder = std::stoi(tokens[9]);
                }
                currentBillboard->layers.push_back(layer);
            }
        }
    }

    file.close();
    return true;
}

