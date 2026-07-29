#include "Math.hpp"
#include <glm/glm.hpp>
#include "BoundingSphere.hpp"
#include "BoundingBox.hpp"
#include <filesystem>
#include "BrushMode.hpp"
#include <sstream>
#include <fstream>
#include <zlib.h>

bool Math::isBetween(float x, float min, float max) {
	return min <= x && x <= max;
}

int Math::clamp(int val, int min, int max) {
	return val < min ? min : val > max ? max : val;
}

float Math::clamp(float val, float min, float max) {
	return val < min ? min : val > max ? max : val;
}

void ensureFolderExists(const std::string& folder) {
    if (!std::filesystem::exists(folder)) {
        std::filesystem::create_directories(folder);
    }
}

std::stringstream gzipDecompressFromIfstream(std::ifstream& inputFile) {
    if (!inputFile) {
        throw std::runtime_error("Failed to open input file.");
    }

    z_stream strm = {};
    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib for decompression.");
    }

    std::stringstream decompressedStream;
    std::vector<char> inBuffer(1024);
    std::vector<char> outBuffer(1024);

    int ret=0;
    do {
        inputFile.read(inBuffer.data(), inBuffer.size());
        strm.next_in = reinterpret_cast<Bytef*>(inBuffer.data());
        strm.avail_in = static_cast<uInt>(inputFile.gcount());

        if (strm.avail_in == 0) {
            break; 
        }

        do {
            strm.next_out = reinterpret_cast<Bytef*>(outBuffer.data());
            strm.avail_out = outBuffer.size();

            ret = inflate(&strm, Z_NO_FLUSH);

            if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
                inflateEnd(&strm);
                throw std::runtime_error("Decompression failed: inflate() error " + std::to_string(ret));
            }

            decompressedStream.write(outBuffer.data(), outBuffer.size() - strm.avail_out);
        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    inflateEnd(&strm);

    if (ret != Z_STREAM_END) {
        throw std::runtime_error("Decompression finished unexpectedly. inflate() error " + std::to_string(ret));
    }

    return decompressedStream;
}

void gzipCompressToOfstream(std::istream& inputStream, std::ofstream& outputFile) {
    if (!outputFile) {
        throw std::runtime_error("Failed to open output file.");
    }

    z_stream strm = {};
    if (deflateInit2(&strm, Z_BEST_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib for compression.");
    }

    std::vector<char> inBuffer(1024);
    std::vector<char> outBuffer(1024);

    int ret;
    do {
        inputStream.read(inBuffer.data(), inBuffer.size());
        strm.next_in = reinterpret_cast<Bytef*>(inBuffer.data());
        strm.avail_in = inputStream.gcount();  // Number of bytes read

        int flush = inputStream.eof() ? Z_FINISH : Z_NO_FLUSH;

        do {
            strm.next_out = reinterpret_cast<Bytef*>(outBuffer.data());
            strm.avail_out = outBuffer.size();

            ret = deflate(&strm, flush);

            if (ret < 0) {
                deflateEnd(&strm);
                throw std::runtime_error("Compression failed: deflate() error " + std::to_string(ret));
            }

            outputFile.write(outBuffer.data(), outBuffer.size() - strm.avail_out);
        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    deflateEnd(&strm);
}

glm::quat Math::eulerToQuat(float yaw, float pitch, float roll) {
    // Convert degrees to radians
    float yawRad = glm::radians(yaw);
    float pitchRad = glm::radians(pitch);
    float rollRad = glm::radians(roll);

    // Construct quaternion in correct order (Yaw -> Pitch -> Roll)
    glm::quat qYaw   = glm::angleAxis(yawRad, glm::vec3(0, 1, 0));  // Rotate around Y
    glm::quat qPitch = glm::angleAxis(pitchRad, glm::vec3(1, 0, 0)); // Rotate around X
    glm::quat qRoll  = glm::angleAxis(rollRad, glm::vec3(0, 0, 1));  // Rotate around Z

    return qYaw * qPitch * qRoll; // Yaw first, then Pitch, then Roll
}

// Generate random float in range [0,1]

float Math::randomFloat() {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

glm::vec3 Math::solveLinearSystem(const glm::mat3& A, const glm::vec3& b) {
    float detA = glm::determinant(A);
    if (std::abs(detA) < 1e-6f) {
        return glm::vec3(0.0f); // Or handle singular matrix
    }

    glm::mat3 A1 = A;
    glm::mat3 A2 = A;
    glm::mat3 A3 = A;

    A1[0] = b;
    A2[1] = b;
    A3[2] = b;

    float x = glm::determinant(A1) / detA;
    float y = glm::determinant(A2) / detA;
    float z = glm::determinant(A3) / detA;

    return glm::vec3(x, y, z);
}

float Math::squaredDistPointAABB(glm::vec3 p, glm::vec3 min, glm::vec3 max){
    float sq = 0.0f;

    sq += Math::check(p[0], min[0], max[0]);
    sq += Math::check(p[1], min[1], max[1]);
    sq += Math::check(p[2], min[2], max[2]);

    return sq;
}

float Math::check(float p, float min, float max){ 
    float out = 0.0f;
    float v = p;

    if (v < min) {             
        float val = (min - v);             
        out += val * val;         
    }         

    if (v > max) {
        float val = (v - max);
        out += val * val;
    }

    return out;
}

const char* toString(BrushMode v)
{
    switch (v)
    {
        case BrushMode::ADD:     return "Add";
        case BrushMode::REMOVE:  return "Remove";
        case BrushMode::REPLACE: return "Replace";
        case BrushMode::PAINT:   return "Paint";
        default:                 return "Unknown";
    }
}

float Math::brightnessAndContrast(float color, float brightness, float contrast) {
    color += brightness;
    color = glm::clamp(color, -1.0f, 1.0f);
    color *= contrast;
    return glm::clamp(color, -1.0f, 1.0f);
}