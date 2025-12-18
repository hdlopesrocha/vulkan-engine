#include "VertexBufferObjectBuilder.hpp"
#include "VulkanApp.hpp"

VertexBufferObject VertexBufferObjectBuilder::create(VulkanApp* app, const Geometry& model) {
    Buffer vb = app->createVertexBuffer(model.vertices);
    Buffer ib = app->createIndexBuffer(model.indices);
    VertexBufferObject vbo{vb, ib, static_cast<uint32_t>(model.indices.size())};
    return vbo;
}
