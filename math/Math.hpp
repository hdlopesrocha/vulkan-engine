#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include "BoundingBox.hpp"
#include "BoundingSphere.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <istream>
#include <iostream>
#include <filesystem>
#include <random>
#include <cmath>

class Math {
public:
    Math();
    ~Math();
    static bool isBetween(float x, float min, float max);
    static int clamp(int val, int min, int max);
    static float clamp(float val, float min, float max);
    static glm::quat eulerToQuat(float yaw, float pitch, float roll);
    static float squaredDistPointAABB(glm::vec3 p, glm::vec3 min, glm::vec3 max);
    static float check(float p, float min, float max);
    static float randomFloat();
    static glm::vec3 solveLinearSystem(const glm::mat3& A, const glm::vec3& b);
    static float brightnessAndContrast(float color, float brightness, float contrast);
};

void ensureFolderExists(const std::string& folder);
std::stringstream gzipDecompressFromIfstream(std::ifstream& inputFile);
void gzipCompressToOfstream(std::istream& inputStream, std::ofstream& outputFile);

 
