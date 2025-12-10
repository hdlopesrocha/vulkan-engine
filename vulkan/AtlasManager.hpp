#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

// Represents a single tile/region in a texture atlas
struct AtlasTile {
    std::string name;
    float offsetX = 0.0f;      // UV offset X (0-1)
    float offsetY = 0.0f;      // UV offset Y (0-1)
    float scaleX = 1.0f;       // UV scale X (0-1)
    float scaleY = 1.0f;       // UV scale Y (0-1)
};

// Manages atlas tile definitions for multiple texture atlases (no ImGui dependency)
// Each atlas is identified by an integer index (e.g., texture index)
class AtlasManager {
public:
    AtlasManager() = default;
    
    // Add a new tile to a specific atlas
    size_t addTile(int atlasIndex, const AtlasTile& tile) {
        atlases[atlasIndex].push_back(tile);
        return atlases[atlasIndex].size() - 1;
    }
    
    // Add a new tile with parameters to a specific atlas
    size_t addTile(int atlasIndex, const std::string& name, float offsetX, float offsetY, float scaleX, float scaleY) {
        AtlasTile tile;
        tile.name = name;
        tile.offsetX = offsetX;
        tile.offsetY = offsetY;
        tile.scaleX = scaleX;
        tile.scaleY = scaleY;
        atlases[atlasIndex].push_back(tile);
        return atlases[atlasIndex].size() - 1;
    }
    
    // Remove a tile by index from a specific atlas
    void removeTile(int atlasIndex, size_t tileIndex) {
        if (atlases.find(atlasIndex) != atlases.end() && tileIndex < atlases[atlasIndex].size()) {
            atlases[atlasIndex].erase(atlases[atlasIndex].begin() + tileIndex);
        }
    }
    
    // Get a tile by index from a specific atlas
    AtlasTile* getTile(int atlasIndex, size_t tileIndex) {
        if (atlases.find(atlasIndex) != atlases.end() && tileIndex < atlases[atlasIndex].size()) {
            return &atlases[atlasIndex][tileIndex];
        }
        return nullptr;
    }
    
    const AtlasTile* getTile(int atlasIndex, size_t tileIndex) const {
        auto it = atlases.find(atlasIndex);
        if (it != atlases.end() && tileIndex < it->second.size()) {
            return &it->second[tileIndex];
        }
        return nullptr;
    }
    
    // Get all tiles for a specific atlas
    const std::vector<AtlasTile>& getTiles(int atlasIndex) const {
        static const std::vector<AtlasTile> empty;
        auto it = atlases.find(atlasIndex);
        if (it != atlases.end()) {
            return it->second;
        }
        return empty;
    }
    
    // Get tile count for a specific atlas
    size_t getTileCount(int atlasIndex) const {
        auto it = atlases.find(atlasIndex);
        if (it != atlases.end()) {
            return it->second.size();
        }
        return 0;
    }
    
    // Clear all tiles for a specific atlas
    void clear(int atlasIndex) {
        atlases[atlasIndex].clear();
    }
    
    // Clear all atlases
    void clearAll() {
        atlases.clear();
    }
    
    // Export tiles to CSV format for a specific atlas
    std::string exportToString(int atlasIndex) const {
        std::string result = "# Atlas Tiles for Atlas " + std::to_string(atlasIndex) + "\n";
        result += "# Format: name,offsetX,offsetY,scaleX,scaleY\n";
        auto it = atlases.find(atlasIndex);
        if (it != atlases.end()) {
            for (const auto& tile : it->second) {
                result += tile.name + "," + 
                          std::to_string(tile.offsetX) + "," +
                          std::to_string(tile.offsetY) + "," +
                          std::to_string(tile.scaleX) + "," +
                          std::to_string(tile.scaleY) + "\n";
            }
        }
        return result;
    }
    
    // Export all atlases to CSV format
    std::string exportAllToString() const {
        std::string result;
        for (const auto& [atlasIndex, tiles] : atlases) {
            result += "# Atlas " + std::to_string(atlasIndex) + "\n";
            result += "# Format: name,offsetX,offsetY,scaleX,scaleY\n";
            for (const auto& tile : tiles) {
                result += std::to_string(atlasIndex) + "," +
                          tile.name + "," + 
                          std::to_string(tile.offsetX) + "," +
                          std::to_string(tile.offsetY) + "," +
                          std::to_string(tile.scaleX) + "," +
                          std::to_string(tile.scaleY) + "\n";
            }
            result += "\n";
        }
        return result;
    }
    
    // Save a specific atlas to file
    bool saveToFile(int atlasIndex, const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        file << exportToString(atlasIndex);
        file.close();
        return true;
    }
    
    // Save all atlases to file
    bool saveAllToFile(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        file << exportAllToString();
        file.close();
        return true;
    }
    
    // Load from file (supports both single atlas and multi-atlas format)
    bool loadFromFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        atlases.clear();
        std::string line;
        int currentAtlasIndex = 0;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // Parse CSV line
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }
            
            // Check if this is multi-atlas format (atlasIndex,name,offsetX,offsetY,scaleX,scaleY)
            // or single atlas format (name,offsetX,offsetY,scaleX,scaleY)
            if (tokens.size() == 6) {
                // Multi-atlas format
                int atlasIdx = std::stoi(tokens[0]);
                std::string name = tokens[1];
                float offsetX = std::stof(tokens[2]);
                float offsetY = std::stof(tokens[3]);
                float scaleX = std::stof(tokens[4]);
                float scaleY = std::stof(tokens[5]);
                addTile(atlasIdx, name, offsetX, offsetY, scaleX, scaleY);
            } else if (tokens.size() == 5) {
                // Single atlas format
                std::string name = tokens[0];
                float offsetX = std::stof(tokens[1]);
                float offsetY = std::stof(tokens[2]);
                float scaleX = std::stof(tokens[3]);
                float scaleY = std::stof(tokens[4]);
                addTile(currentAtlasIndex, name, offsetX, offsetY, scaleX, scaleY);
            }
        }
        
        file.close();
        return true;
    }
    
private:
    // Map from atlas index to list of tiles for that atlas
    std::map<int, std::vector<AtlasTile>> atlases;
};
