#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <set>
// implementation-heavy functions are defined in AtlasManager.cpp

#include "AtlasTile.hpp"

// Manages atlas tile definitions for multiple texture atlases (no ImGui dependency)
// Each atlas is identified by an integer index (e.g., texture index)
class AtlasManager {
public:
    AtlasManager() = default;
    
    // Add a new tile to a specific atlas
    size_t addTile(int atlasIndex, const AtlasTile& tile);
    
    // Add a new tile with parameters to a specific atlas
    size_t addTile(int atlasIndex, const std::string& name, float offsetX, float offsetY, float scaleX, float scaleY);
    
    // Remove a tile by index from a specific atlas
    void removeTile(int atlasIndex, size_t tileIndex);
    
    // Get a tile by index from a specific atlas
    AtlasTile* getTile(int atlasIndex, size_t tileIndex);
    
    const AtlasTile* getTile(int atlasIndex, size_t tileIndex) const;
    
    // Get all tiles for a specific atlas
    const std::vector<AtlasTile>& getTiles(int atlasIndex) const;
    
    // Get tile count for a specific atlas
    size_t getTileCount(int atlasIndex) const;
    
    // Clear all tiles for a specific atlas
    void clear(int atlasIndex);
    
    // Clear all atlases
    void clearAll();
    
    // Export tiles to CSV format for a specific atlas
    std::string exportToString(int atlasIndex) const;
    
    // Export all atlases to CSV format
    std::string exportAllToString() const;
    
    // Save a specific atlas to file
    bool saveToFile(int atlasIndex, const std::string& filepath) const;
    
    // Save all atlases to file
    bool saveAllToFile(const std::string& filepath) const;
    
    // Load from file (supports both single atlas and multi-atlas format)
    bool loadFromFile(const std::string& filepath);
    
    // Auto-detect tiles from an opacity/alpha map image
    // Returns the number of tiles detected and added
    int autoDetectTiles(int atlasIndex, const std::string& opacityImagePath, int threshold = 10);
    
private:
    // Map from atlas index to list of tiles for that atlas
    std::map<int, std::vector<AtlasTile>> atlases;
    
    // Helper function for flood fill to find bounding box
    void floodFillBounds(unsigned char* imageData, std::vector<std::vector<bool>>& visited, 
                         int width, int height, int startX, int startY, int threshold,
                         int& minX, int& maxX, int& minY, int& maxY);
};
