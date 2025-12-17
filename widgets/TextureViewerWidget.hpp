#pragma once

#include <imgui.h>
#include <functional>
#include "../vulkan/TextureArrayManager.hpp"
#include "../vulkan/MaterialManager.hpp"
#include "../utils/MaterialProperties.hpp"
#include "Widget.hpp"

class TextureViewer : public Widget {
public:
    TextureViewer();
    void init(TextureArrayManager* arrayManager, std::vector<MaterialProperties>* materials);
    void render() override;
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = cb; }

private:
    TextureArrayManager* arrayManager = nullptr;
    std::vector<MaterialProperties>* materials = nullptr;
    size_t currentIndex = 0;
    std::function<void(size_t)> onMaterialChanged;
};
