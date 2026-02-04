#include "TexturePreview.hpp"
#include <imgui.h>
#include <algorithm>

void ImGuiComponents::RenderTexturePreview(ImTextureID texID, float previewSize, uint32_t width, uint32_t height, const char* format) {
    ImVec2 imageSize(previewSize, previewSize);
    if (texID) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 end = ImVec2(pos.x + imageSize.x, pos.y + imageSize.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float tile = 16.0f;
        ImU32 colA = IM_COL32(200,200,200,255);
        ImU32 colB = IM_COL32(160,160,160,255);
        for (float y = pos.y; y < end.y; y += tile) {
            for (float x = pos.x; x < end.x; x += tile) {
                bool odd = (static_cast<int>((x - pos.x) / tile) + static_cast<int>((y - pos.y) / tile)) & 1;
                ImVec2 q0(x, y);
                ImVec2 q1(std::min(x + tile, end.x), std::min(y + tile, end.y));
                dl->AddRectFilled(q0, q1, odd ? colA : colB);
            }
        }
        ImGui::Image(texID, imageSize);
    } else {
        ImGui::Text("Texture preview not available");
    }

    ImGui::Text("Size: %dx%d", width, height);
    ImGui::Text("Format: %s", format ? format : "Unknown");
}