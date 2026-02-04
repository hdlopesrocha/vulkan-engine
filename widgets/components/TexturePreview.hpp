#pragma once

#include <imgui.h>
#include <cstdint>

namespace ImGuiComponents {
    // Renders a square texture preview with checkerboard background and size/format info.
    // - texID: ImGui texture handle (can be nullptr)
    // - previewSize: side length in pixels
    // - width/height: reported texture resolution
    // - format: human-readable format string (e.g., "RGBA8")
    void RenderTexturePreview(ImTextureID texID, float previewSize, uint32_t width, uint32_t height, const char* format);
}