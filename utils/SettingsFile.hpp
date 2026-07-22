#pragma once
#include "../utils/Settings.hpp"
#include <string>

class SettingsFile {
    Settings* settings;
    std::string filename;
public:
    SettingsFile(Settings* settings_, std::string filename_);
    void save(std::string baseFolder);
    void load(std::string baseFolder);
};
