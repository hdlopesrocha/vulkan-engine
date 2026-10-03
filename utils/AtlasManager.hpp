#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <set>

#include "AtlasTile.hpp"

// Manages atlas tile definitions for multiple texture atlases (no ImGui dependency)
// Each atlas is identified by an integer index (e.g., texture index)
class AtlasManager {
public:
    AtlasManager() = default;
    
    // Add a new tile to a specific atlas
    size_t addTile(int atlasIndex, const AtlasTile& tile);
    
    // Remove a tile by index from a specific atlas
    void removeTile(int atlasIndex, size_t tileIndex);
    
    // Get a tile by index from a specific atlas
    AtlasTile* getTile(int atlasIndex, size_t tileIndex);
    
    const AtlasTile* getTile(int atlasIndex, size_t tileIndex) const;
    
    // Get tile count for a specific atlas
    size_t getTileCount(int atlasIndex) const;
    
    // Clear all tiles for a specific atlas
    void clear(int atlasIndex);
    
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
