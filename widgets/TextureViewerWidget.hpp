#pragma once

#include <imgui.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>
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
    void setVideoSlotNamesProvider(std::function<std::vector<std::string>()> fn) { videoSlotNamesProvider = std::move(fn); }

private:
    TextureArrayManager* arrayManager = nullptr;
    std::vector<MaterialProperties>* materials = nullptr;
    size_t currentIndex = 0;
    std::function<void(size_t)> onMaterialChanged;
    std::function<std::vector<std::string>()> videoSlotNamesProvider;
};
