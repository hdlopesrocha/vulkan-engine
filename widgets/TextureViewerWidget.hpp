#pragma once

#include <imgui.h>
#include <functional>
#include "../vulkan/TextureManager.hpp"
#include "Widget.hpp"

class TextureViewer : public Widget {
public:
    TextureViewer();
    void init(TextureManager* manager);
    void render() override;
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = cb; }

private:
    TextureManager* manager = nullptr;
    size_t currentIndex = 0;
    std::function<void(size_t)> onMaterialChanged;
};
