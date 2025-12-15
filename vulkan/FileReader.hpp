#pragma once

#include <string>
#include <vector>

class FileReader {
public:
    static std::vector<char> readFile(const std::string& filename);
};
