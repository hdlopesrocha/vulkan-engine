#pragma once

#include "Vertex.hpp"

struct VertexHasher {
   uint64_t operator()(const Vertex& v) const noexcept {
        uint64_t h = 0;
        h = hashCombine(h, std::hash<glm::vec3>{}(v.position));
        h = hashCombine(h, std::hash<glm::vec3>{}(v.normal));
        h = hashCombine(h, std::hash<glm::vec2>{}(v.texCoord));
        h = hashCombine(h, std::hash<int>{}(v.texIndex));
        return h;
    }
};
