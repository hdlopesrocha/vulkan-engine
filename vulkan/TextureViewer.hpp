#pragma once

#include <imgui.h>
#include "TextureManager.hpp"

class TextureViewer {
public:
    TextureViewer() = default;
    void init(TextureManager* manager) { this->manager = manager; }
    void render();

private:
    TextureManager* manager = nullptr;
    size_t currentIndex = 0;
};
