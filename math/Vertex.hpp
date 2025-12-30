#pragma once
#include <glm/glm.hpp>
#include <bit>
#include <cstdint>
#include <utility>
#include <array>
#include <tuple>
#include <functional>

inline uint64_t murmurMix(uint64_t k) {
	k ^= k >> 33;
	k *= 0xff51afd7ed558ccdULL;
	k ^= k >> 33;
	k *= 0xc4ceb9fe1a85ec53ULL;
	k ^= k >> 33;
	return k;
}

inline uint64_t hashCombine(uint64_t h, uint64_t v) {
	return h ^ murmurMix(v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

inline uint64_t pack2(float a, float b) {
	struct Pair { float x; float y; };
	static_assert(sizeof(Pair) == sizeof(uint64_t));
	return std::bit_cast<uint64_t>(Pair{a, b});
}

struct alignas(16) Vertex {
public:
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;
    glm::vec4 tangent; // xyz = tangent, w = handedness
    int texIndex;
    int _pad0;

    Vertex(glm::vec3 pos, glm::vec3 normal, glm::vec2 texCoord, int texIndex)
        : position(pos), color(glm::vec3(1.0f)), texCoord(texCoord), normal(normal), tangent(0.0f,0.0f,0.0f,1.0f), texIndex(texIndex), _pad0(0) {
    }

    // Compatibility constructor to allow aggregate-style initialization used across the codebase
    Vertex(std::array<float,3> posArr, std::array<float,3> colorArr, std::array<float,2> texArr, std::array<float,3> normalArr, float texIndexF)
        : position(posArr[0], posArr[1], posArr[2]),
          color(colorArr[0], colorArr[1], colorArr[2]),
          texCoord(texArr[0], texArr[1]),
          normal(normalArr[0], normalArr[1], normalArr[2]),
          tangent(0.0f,0.0f,0.0f,1.0f),
          texIndex(static_cast<int>(texIndexF)), _pad0(0) {}

    Vertex() : position(glm::vec3(0.0f)), color(glm::vec3(1.0f)), texCoord(glm::vec2(0.0f)), normal(glm::vec3(0.0f)), tangent(0.0f,0.0f,0.0f,1.0f), texIndex(0), _pad0(0) {}

    Vertex(glm::vec3 pos) : position(pos), color(glm::vec3(1.0f)), texCoord(glm::vec2(0.0f)), normal(glm::vec3(0.0f)), tangent(0.0f,0.0f,0.0f,1.0f), texIndex(0), _pad0(0) {}

    bool operator<(const Vertex& other) const {
        return std::tie(position.x, position.y, position.z, normal.x, normal.y, normal.z, texCoord.x, texCoord.y, texIndex, tangent.x, tangent.y)
             < std::tie(other.position.x, other.position.y, other.position.z, other.normal.x, other.normal.y, other.normal.z, other.texCoord.x, other.texCoord.y, other.texIndex, other.tangent.x, other.tangent.y);
    }

    bool operator==(const Vertex& o) const {
        return std::bit_cast<uint64_t>(pack2(position.x, position.y)) == std::bit_cast<uint64_t>(pack2(o.position.x, o.position.y)) &&
               std::bit_cast<uint32_t>(position.z) == std::bit_cast<uint32_t>(o.position.z) &&

               std::bit_cast<uint64_t>(pack2(normal.x, normal.y)) == std::bit_cast<uint64_t>(pack2(o.normal.x, o.normal.y)) &&
               std::bit_cast<uint32_t>(normal.z) == std::bit_cast<uint32_t>(o.normal.z) &&

               std::bit_cast<uint64_t>(pack2(texCoord.x, texCoord.y)) == std::bit_cast<uint64_t>(pack2(o.texCoord.x, o.texCoord.y)) &&

               std::bit_cast<uint64_t>(pack2(tangent.x, tangent.y)) == std::bit_cast<uint64_t>(pack2(o.tangent.x, o.tangent.y)) &&
               std::bit_cast<uint32_t>(tangent.z) == std::bit_cast<uint32_t>(o.tangent.z) &&
               std::bit_cast<uint32_t>(tangent.w) == std::bit_cast<uint32_t>(o.tangent.w) &&

               texIndex == o.texIndex;
    }

    bool operator!=(const Vertex& other) const {
        return !(*this == other);
    }
};

namespace std {

template <> struct hash<glm::vec4> {
    uint64_t operator()(const glm::vec4& v) const {
        uint64_t h = 0;
        h = hashCombine(h, pack2(v.x, v.y));
        h = hashCombine(h, pack2(v.z, v.w));
        return h;
    }
};

template<> struct hash<glm::vec3> {
    uint64_t operator()(const glm::vec3& v) const noexcept {
        uint64_t h = 0;
        h = hashCombine(h, pack2(v.x, v.y));
        uint64_t zbits = std::bit_cast<uint32_t>(v.z);
        h = hashCombine(h, zbits);
        return h;
    }
};

template<> struct hash<glm::vec2> {
    uint64_t operator()(const glm::vec2& v) const noexcept {
        uint64_t k = pack2(v.x, v.y);
        return murmurMix(k);
    }
};

} // namespace std


 
