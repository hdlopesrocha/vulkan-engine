#pragma once

#include "Widget.hpp"
#include <memory>
#include "../vulkan/EditableTextureSet.hpp"

class AnimatedTextureWidget : public Widget {
public:
    AnimatedTextureWidget(std::shared_ptr<EditableTextureSet> textures, const char* title = "Editable Textures");
    void render() override;

private:
    std::shared_ptr<EditableTextureSet> textures;

    // UI state for Perlin parameters
    int perlinScale = 8;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;
    float perlinContrast = 5.0f;
    float perlinTime = 0.0f;
    unsigned int perlinSeed = 12345u;

    void renderTextureTab(EditableTexture& texture, int map);
};