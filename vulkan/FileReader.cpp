#include "vulkan.hpp"
#include "FileReader.hpp"


std::vector<char> FileReader::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        // Fallback: try loading from build output directory (bin/) if present
        std::string alt = std::string("bin/") + filename;
        std::ifstream altFile(alt, std::ios::ate | std::ios::binary);
        if (altFile.is_open()) {
            size_t fileSize = (size_t)altFile.tellg();
            std::vector<char> buffer(fileSize);
            altFile.seekg(0);
            altFile.read(buffer.data(), fileSize);
            altFile.close();
            return buffer;
        }
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}