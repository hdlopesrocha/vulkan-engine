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
    static int triplanarPlane(glm::vec3 position, glm::vec3 normal);
    static int mod(int a, int b);
    static glm::vec2 triplanarMapping(glm::vec3 position, int plane);
    static glm::vec3 surfaceNormal(const glm::vec3 point, const BoundingBox &box);
    static glm::vec3 surfaceNormal(const glm::vec3 point, const BoundingSphere &sphere);
    static glm::mat4 getCanonicalMVP(glm::mat4 m);
    static glm::mat4 getRotationMatrixFromNormal(glm::vec3 normal, glm::vec3 target);
    static float triangleArea(const glm::vec3& A, const glm::vec3& B, const glm::vec3& C);
    static double degToRad(double degrees);
    static void wgs84ToEcef(double lat, double lon, double height, double &X, double &Y, double &Z);
    static glm::quat createQuaternion(float yaw, float pitch, float roll);
    static glm::quat eulerToQuat(float yaw, float pitch, float roll);
    static float squaredDistPointAABB(glm::vec3 p, glm::vec3 min, glm::vec3 max);
    static float check(float p, float min, float max);
    static float randomFloat();
    static glm::vec3 solveLinearSystem(const glm::mat3& A, const glm::vec3& b);
    static float brightnessAndContrast(float color, float brightness, float contrast);
    static glm::vec3 hsv2rgb(const glm::vec3& c);
    static glm::vec3 brushColor(unsigned int i);
};

void ensureFolderExists(const std::string& folder);
std::stringstream gzipDecompressFromIfstream(std::ifstream& inputFile);
void gzipCompressToOfstream(std::istream& inputStream, std::ofstream& outputFile);

 
