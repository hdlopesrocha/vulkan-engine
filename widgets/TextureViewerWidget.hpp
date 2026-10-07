#pragma once

#include <imgui.h>
#include <functional>
#include "../vulkan/resources/TextureArrayManager.hpp"
#include "../vulkan/resources/MaterialManager.hpp"
#include "types/MaterialProperties.hpp"
#include "Widget.hpp"

class TextureViewer : public Widget {
public:
    TextureViewer();
    void init(TextureArrayManager* arrayManager_, std::vector<MaterialProperties>* materials_);
    void render() override;
    void setOnMaterialChanged(std::function<void(size_t)> cb) { onMaterialChanged = cb; }

private:
    TextureArrayManager* arrayManager = nullptr;
    std::vector<MaterialProperties>* materials = nullptr;
    size_t currentIndex = 0;
    std::function<void(size_t)> onMaterialChanged;
};
