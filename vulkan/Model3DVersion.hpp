#pragma once

// Simple versioned mesh handle used by main.cpp when models are loaded asynchronously.
// We store a compact mesh id (managed by IndirectRenderer) instead of owning a Model3D pointer.
struct Model3DVersion
{
    uint32_t meshId = UINT32_MAX;
    uint version = 0;
};
