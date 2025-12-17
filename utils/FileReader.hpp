#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class FileReader {
public:
    static std::vector<char> readFile(const std::string& filename);
};
