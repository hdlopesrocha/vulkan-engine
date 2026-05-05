#include "SettingsFile.hpp"
#include "../math/Math.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

SettingsFile::SettingsFile(Settings* settings, std::string filename) {
    this->settings = settings;
    this->filename = filename;
}

void SettingsFile::load(std::string baseFolder) {
    std::string filePath = baseFolder + "/" + filename + ".bin";
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "SettingsFile::load() Error opening file: " << filePath << std::endl;
        return;
    }
    std::stringstream decompressed = gzipDecompressFromIfstream(file);
    decompressed.read(reinterpret_cast<char*>(settings), sizeof(Settings));
    file.close();
    std::cout << "SettingsFile::load('" << filePath << "') Ok!" << std::endl;
}

void SettingsFile::save(std::string baseFolder) {
    ensureFolderExists(baseFolder);
    std::string filePath = baseFolder + "/" + filename + ".bin";
    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "SettingsFile::save() Error opening file: " << filePath << std::endl;
        return;
    }
    std::ostringstream buf;
    buf.write(reinterpret_cast<const char*>(settings), sizeof(Settings));
    std::istringstream inputStream(buf.str());
    gzipCompressToOfstream(inputStream, file);
    file.close();
    std::cout << "SettingsFile::save('" << filePath << "') Ok!" << std::endl;
}
