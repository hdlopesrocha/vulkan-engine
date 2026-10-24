#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// One chunk's triangle span fed to the RT scene BLAS.
struct RTGeometrySpan {
    uint32_t chunkId = 0;      // NodeID (joins the proxy material registry)
    uint32_t baseVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t level = 0;         // 0-based LoD rung (0 = finest published rung)
    glm::vec3 spanMin = glm::vec3(0.0f); // emitting cell bounds (nesting test)
    glm::vec3 spanMax = glm::vec3(0.0f);
    bool operator==(const RTGeometrySpan& o) const {
        return chunkId == o.chunkId && baseVertex == o.baseVertex &&
               vertexCount == o.vertexCount && firstIndex == o.firstIndex &&
               indexCount == o.indexCount && level == o.level &&
               spanMin == o.spanMin && spanMax == o.spanMax;
    }
};
