#pragma once

#include <imgui.h>
#include "../vulkan/TextureManager.hpp"
#include "Widget.hpp"

class TextureViewer : public Widget {
public:
    TextureViewer();
    void init(TextureManager* manager);
    void render() override;

private:
    TextureManager* manager = nullptr;
    size_t currentIndex = 0;
};
