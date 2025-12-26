#include "IndirectRenderer.hpp"
#include "VulkanApp.hpp"

IndirectRenderer::IndirectRenderer() {}
IndirectRenderer::~IndirectRenderer() {}

void IndirectRenderer::init(VulkanApp* app) {
    (void)app;
}

void IndirectRenderer::cleanup(VulkanApp* app) {
    // Destroy per-mesh buffers
    for (auto &m : meshes) {
        if (m.vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), m.vertexBuffer.buffer, nullptr);
        if (m.vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), m.vertexBuffer.memory, nullptr);
        if (m.indexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), m.indexBuffer.buffer, nullptr);
        if (m.indexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), m.indexBuffer.memory, nullptr);
    }

    if (vertexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), vertexBuffer.buffer, nullptr);
    if (vertexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), vertexBuffer.memory, nullptr);
    if (indexBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), indexBuffer.buffer, nullptr);
    if (indexBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), indexBuffer.memory, nullptr);
    if (indirectBuffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), indirectBuffer.buffer, nullptr);
    if (indirectBuffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), indirectBuffer.memory, nullptr);
}

uint32_t IndirectRenderer::addMesh(VulkanApp* app, const Geometry& mesh, const glm::mat4& model) {
    MeshInfo info;
    info.id = nextId++;
    info.baseVertex = 0;
    info.firstIndex = 0;
    info.indexCount = static_cast<uint32_t>(mesh.indices.size());
    info.model = model;
    info.active = true;

    // Create per-mesh GPU buffers immediately for compatibility with current rendering flow.
    if (mesh.vertices.size() > 0) {
        info.vertexBuffer = app->createVertexBuffer(mesh.vertices);
    }
    if (mesh.indices.size() > 0) {
        info.indexBuffer = app->createIndexBuffer(mesh.indices);
    }

    // Append an indirect command placeholder for this mesh. We set firstIndex/vertexOffset to 0
    // because each mesh uses its own index/vertex buffers; the indirect command uses these buffers
    // when the draw is executed after binding them.
    VkDrawIndexedIndirectCommand cmd{};
    cmd.indexCount = info.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex = 0;
    cmd.vertexOffset = 0;
    cmd.firstInstance = 0;

    idToIndex[info.id] = meshes.size();
    meshes.push_back(info);

    // stash command and upload indirect buffer
    indirectCommands.push_back(cmd);
    dirty = true;
    // immediate rebuild so indirectBuffer is created
    rebuild(app);

    return info.id;
}

void IndirectRenderer::removeMesh(uint32_t meshId) {
    auto it = idToIndex.find(meshId);
    if (it == idToIndex.end()) return;
    size_t idx = it->second;
    meshes[idx].active = false;
    idToIndex.erase(it);
    dirty = true;
}

void IndirectRenderer::rebuild(VulkanApp* app) {
    if (!dirty) return;

    // Rebuild merged vertex/index/indirect arrays from active meshes
    mergedVertices.clear();
    mergedIndices.clear();
    indirectCommands.clear();

    // We'll need access to the original geometry for index/vertex data; the app
    // currently passes Geometry to addMesh but we did not store it to avoid copies.
    // To keep this implementation simple and safe, we expect the caller to have
    // retained geometry externally and call rebuild immediately after addMesh.
    // As a fallback, we'll build empty commands for stored meshes (indexCount only).

    uint32_t vertexCursor = 0;
    VkDeviceSize indirectOffset = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        auto &m = meshes[i];
        if (!m.active) continue;

        // Create an indirect draw command placeholder. The app will supply indexCount
        // values when adding; we use stored indexCount here but indices/vertices must
        // be provided externally. For now, index/vertex buffers will remain empty
        // until the app uploads actual data via createVertexBuffer/createIndexBuffer.
        VkDrawIndexedIndirectCommand cmd{};
        cmd.indexCount = m.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = 0; // when mergedIndices are built these will be updated
        cmd.vertexOffset = 0; // updated later
        cmd.firstInstance = 0;

        // store offsets (byte offset into indirect buffer)
        m.indirectOffset = indirectOffset;
        indirectOffset += sizeof(VkDrawIndexedIndirectCommand);

        indirectCommands.push_back(cmd);
    }

    // Create or recreate GPU indirect buffer and upload commands
    VkDeviceSize indirectSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
    if (indirectBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), indirectBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), indirectBuffer.memory, nullptr);
        indirectBuffer = {};
    }
    if (indirectSize > 0) {
        indirectBuffer = app->createBuffer(indirectSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* data;
        vkMapMemory(app->getDevice(), indirectBuffer.memory, 0, indirectSize, 0, &data);
        memcpy(data, indirectCommands.data(), (size_t)indirectSize);
        vkUnmapMemory(app->getDevice(), indirectBuffer.memory);
    }

    // Note: this simple implementation does not perform full vertex/index merging
    // from Geometry objects; it focuses on managing indirect commands and offsets.
    // A more advanced allocator would accept Geometry data and rebuild merged
    // vertex/index buffers here.

    dirty = false;
}

void IndirectRenderer::drawIndirect(VkCommandBuffer cmd, uint32_t meshId) {
    auto it = idToIndex.find(meshId);
    if (it == idToIndex.end()) return;
    const MeshInfo &m = meshes[it->second];
    if (!m.active) return;
    if (indirectBuffer.buffer == VK_NULL_HANDLE) return;

    vkCmdDrawIndexedIndirect(cmd, indirectBuffer.buffer, m.indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
}

IndirectRenderer::MeshInfo IndirectRenderer::getMeshInfo(uint32_t meshId) const {
    IndirectRenderer::MeshInfo empty;
    auto it = idToIndex.find(meshId);
    if (it == idToIndex.end()) return empty;
    return meshes[it->second];
}
 