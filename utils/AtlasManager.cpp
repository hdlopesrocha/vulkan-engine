#include "AtlasManager.hpp"
#include <stb/stb_image.h>

size_t AtlasManager::addTile(int atlasIndex, const AtlasTile& tile) {
    atlases[atlasIndex].push_back(tile);
    return atlases[atlasIndex].size() - 1;
}

void AtlasManager::removeTile(int atlasIndex, size_t tileIndex) {
    if (atlases.find(atlasIndex) != atlases.end() && tileIndex < atlases[atlasIndex].size()) {
        atlases[atlasIndex].erase(atlases[atlasIndex].begin() + tileIndex);
    }
}

AtlasTile* AtlasManager::getTile(int atlasIndex, size_t tileIndex) {
    if (atlases.find(atlasIndex) != atlases.end() && tileIndex < atlases[atlasIndex].size()) {
        return &atlases[atlasIndex][tileIndex];
    }
    return nullptr;
}

const AtlasTile* AtlasManager::getTile(int atlasIndex, size_t tileIndex) const {
    auto it = atlases.find(atlasIndex);
    if (it != atlases.end() && tileIndex < it->second.size()) {
        return &it->second[tileIndex];
    }
    return nullptr;
}

size_t AtlasManager::getTileCount(int atlasIndex) const {
    auto it = atlases.find(atlasIndex);
    if (it != atlases.end()) {
        return it->second.size();
    }
    return 0;
}

void AtlasManager::clear(int atlasIndex) {
    atlases[atlasIndex].clear();
}

int AtlasManager::autoDetectTiles(int atlasIndex, const std::string& opacityImagePath, int threshold) {
    int width, height, channels;
    unsigned char* imageData = stbi_load(opacityImagePath.c_str(), &width, &height, &channels, 1); // Load as grayscale

    if (!imageData) {
        return 0;
    }
    
    int tilesAdded = 0;
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
    
    // Find connected regions of non-transparent pixels
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            
            int pixelValue = imageData[y * width + x];
            
            // If pixel is opaque enough and not visited, start a new region
            if (pixelValue > threshold && !visited[y][x]) {
                // Find bounding box of this connected region using flood fill
                int minX = x, maxX = x, minY = y, maxY = y;
                floodFillBounds(imageData, visited, width, height, x, y, threshold, minX, maxX, minY, maxY);
                
                // Skip very small regions (likely noise)
                int regionWidth = maxX - minX + 1;
                int regionHeight = maxY - minY + 1;
                if (regionWidth < 5 || regionHeight < 5) {
                    continue;
                }
                
                // Add some padding to the bounding box (5% of texture size)
                float paddingX = 0.01f;
                float paddingY = 0.01f;
                
                // Create a tile from this bounding box
                AtlasTile tile;
                tile.name = "Auto " + std::to_string(tilesAdded + 1);
                tile.offsetX = std::max(0.0f, static_cast<float>(minX) / width - paddingX);
                tile.offsetY = std::max(0.0f, static_cast<float>(minY) / height - paddingY);
                tile.scaleX = std::min(1.0f - tile.offsetX, static_cast<float>(regionWidth) / width + 2 * paddingX);
                tile.scaleY = std::min(1.0f - tile.offsetY, static_cast<float>(regionHeight) / height + 2 * paddingY);
                
                addTile(atlasIndex, tile);
                tilesAdded++;
            }
        }
    }
    
    stbi_image_free(imageData);
    return tilesAdded;
}

void AtlasManager::floodFillBounds(unsigned char* imageData, std::vector<std::vector<bool>>& visited, 
				 int width, int height, int startX, int startY, int threshold,
				 int& minX, int& maxX, int& minY, int& maxY) {
	std::vector<std::pair<int, int>> stack;
	stack.reserve(1000); // Reserve space to avoid reallocations
	stack.push_back({startX, startY});
	
	while (!stack.empty()) {
		auto [x, y] = stack.back();
		stack.pop_back();
		
		// Check bounds
		if (x < 0 || x >= width || y < 0 || y >= height) continue;
		
		// Check if already visited
		if (visited[y][x]) continue;
		
		// Check pixel value
		int pixelIndex = y * width + x;
		if (pixelIndex < 0 || pixelIndex >= width * height) continue;
		
		int pixelValue = imageData[pixelIndex];
		if (pixelValue <= threshold) continue;
		
		// Mark as visited
		visited[y][x] = true;
		
		// Update bounding box
		minX = std::min(minX, x);
		maxX = std::max(maxX, x);
		minY = std::min(minY, y);
		maxY = std::max(maxY, y);
		
		// Add neighbors to stack (with bounds checking)
		if (x + 1 < width) stack.push_back({x + 1, y});
		if (x - 1 >= 0) stack.push_back({x - 1, y});
		if (y + 1 < height) stack.push_back({x, y + 1});
		if (y - 1 >= 0) stack.push_back({x, y - 1});
	}
}
