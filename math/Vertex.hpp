#ifndef MATH_VERTEX_HPP
#define MATH_VERTEX_HPP

#include <glm/glm.hpp>
#include <bit>
#include <cstdint>
#include <utility>
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
    glm::vec4 tangent;
    int texIndex;
    int brushIndex;
    int _pad0;

    Vertex(glm::vec3 pos, glm::vec3 normal, glm::vec2 texCoord, int brushIndex)
        : position(pos), color(glm::vec3(1.0f)), texCoord(texCoord), normal(normal), texIndex(0), brushIndex(brushIndex), tangent(glm::vec4(0.0f)), _pad0(0) {
    }

    Vertex() : position(glm::vec3(0.0f)), color(glm::vec3(1.0f)), texCoord(glm::vec2(0.0f)), normal(glm::vec3(0.0f)), texIndex(0), brushIndex(0), tangent(glm::vec4(0.0f)), _pad0(0) {}

    Vertex(glm::vec3 pos) : position(pos), color(glm::vec3(1.0f)), texCoord(glm::vec2(0.0f)), normal(glm::vec3(0.0f)), texIndex(0), brushIndex(0), tangent(glm::vec4(0.0f)), _pad0(0) {}

    bool operator<(const Vertex& other) const {
        return std::tie(position.x, position.y, position.z, normal.x, normal.y, normal.z, texCoord.x, texCoord.y, texIndex)
             < std::tie(other.position.x, other.position.y, other.position.z, other.normal.x, other.normal.y, other.normal.z, other.texCoord.x, other.texCoord.y, other.texIndex);
    }

    bool operator==(const Vertex& o) const {
        return std::bit_cast<uint64_t>(pack2(position.x, position.y)) == std::bit_cast<uint64_t>(pack2(o.position.x, o.position.y)) &&
               std::bit_cast<uint32_t>(position.z) == std::bit_cast<uint32_t>(o.position.z) &&

               std::bit_cast<uint64_t>(pack2(normal.x, normal.y)) == std::bit_cast<uint64_t>(pack2(o.normal.x, o.normal.y)) &&
               std::bit_cast<uint32_t>(normal.z) == std::bit_cast<uint32_t>(o.normal.z) &&

               std::bit_cast<uint64_t>(pack2(texCoord.x, texCoord.y)) == std::bit_cast<uint64_t>(pack2(o.texCoord.x, o.texCoord.y)) &&

               texIndex == o.texIndex && brushIndex == o.brushIndex;
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

struct VertexHasher {
   uint64_t operator()(const Vertex& v) const noexcept {
        uint64_t h = 0;
        h = hashCombine(h, std::hash<glm::vec3>{}(v.position));
        h = hashCombine(h, std::hash<glm::vec3>{}(v.normal));
        h = hashCombine(h, std::hash<glm::vec2>{}(v.texCoord));
        h = hashCombine(h, std::hash<int>{}(v.texIndex));
        h = hashCombine(h, std::hash<int>{}(v.brushIndex));
        return h;
    }
};

#endif // MATH_VERTEX_HPP
