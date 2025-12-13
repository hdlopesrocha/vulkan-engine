#include "VertexBufferObjectBuilder.hpp"

VertexBufferObject VertexBufferObjectBuilder::create(VulkanApp* app, const Model3D& model) {
    // Clone vertices and indices so we can call VulkanApp methods
    std::vector<Vertex> vertices = model.getVertices();
    std::vector<uint16_t> indices = model.getIndices();

    Buffer vb = app->createVertexBuffer(vertices);
    Buffer ib = app->createIndexBuffer(indices);
    VertexBufferObject vbo{vb, ib, static_cast<uint32_t>(indices.size())};
    return vbo;
}
