#pragma once

#include <imgui.h>
#include "TextureManager.hpp"
#include "../widgets/Widget.hpp"

class TextureViewer : public Widget {
public:
    TextureViewer() : Widget("Textures") {}
    void init(TextureManager* manager) { this->manager = manager; }
    void render() override;

private:
    TextureManager* manager = nullptr;
    size_t currentIndex = 0;
};
