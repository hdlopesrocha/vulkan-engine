#pragma once

#include "Buffer.hpp"

struct VertexBufferObject {
    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;
};
