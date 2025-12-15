#include "AtlasManager.hpp"
#include <stb/stb_image.h>

size_t AtlasManager::addTile(int atlasIndex, const AtlasTile& tile) {
	atlases[atlasIndex].push_back(tile);
	return atlases[atlasIndex].size() - 1;
}

size_t AtlasManager::addTile(int atlasIndex, const std::string& name, float offsetX, float offsetY, float scaleX, float scaleY) {
	AtlasTile tile;
	tile.name = name;
	tile.offsetX = offsetX;
	tile.offsetY = offsetY;
	tile.scaleX = scaleX;
	tile.scaleY = scaleY;
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

const std::vector<AtlasTile>& AtlasManager::getTiles(int atlasIndex) const {
	static const std::vector<AtlasTile> empty;
	auto it = atlases.find(atlasIndex);
	if (it != atlases.end()) {
		return it->second;
	}
	return empty;
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

void AtlasManager::clearAll() {
	atlases.clear();
}

std::string AtlasManager::exportToString(int atlasIndex) const {
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

std::string AtlasManager::exportAllToString() const {
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

bool AtlasManager::saveToFile(int atlasIndex, const std::string& filepath) const {
	std::ofstream file(filepath);
	if (!file.is_open()) {
		return false;
	}
	file << exportToString(atlasIndex);
	file.close();
	return true;
}

bool AtlasManager::saveAllToFile(const std::string& filepath) const {
	std::ofstream file(filepath);
	if (!file.is_open()) {
		return false;
	}
	file << exportAllToString();
	file.close();
	return true;
}

bool AtlasManager::loadFromFile(const std::string& filepath) {
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
