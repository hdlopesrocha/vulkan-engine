#pragma once

#include "vulkan.hpp"
#include "TextureImage.hpp"
#include <imgui.h>
#include "../utils/MaterialProperties.hpp"

struct Triple {
    TextureImage albedo;
    VkSampler albedoSampler = VK_NULL_HANDLE;
    TextureImage normal;
    VkSampler normalSampler = VK_NULL_HANDLE;
    TextureImage height;
    VkSampler heightSampler = VK_NULL_HANDLE;
    ImTextureID albedoTexID = nullptr;
    ImTextureID normalTexID = nullptr;
    ImTextureID heightTexID = nullptr;
    MaterialProperties material;
    bool ownsResources = true;
};
