#pragma once

// Lightweight holder that tracks a mesh id produced by the IndirectRenderer
// and the source node version. We no longer store heap-allocated Model3D
// instances here; the IndirectRenderer packs geometry into merged buffers.
struct Model3DVersion {
    uint32_t meshId = UINT32_MAX; // stable slot returned by IndirectRenderer::addMeshSlotted
    uint version = 0;
};
