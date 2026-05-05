#pragma once
#include "../widgets/Settings.hpp"
#include <string>

class SettingsFile {
    Settings* settings;
    std::string filename;
public:
    SettingsFile(Settings* settings, std::string filename);
    void save(std::string baseFolder);
    void load(std::string baseFolder);
};
