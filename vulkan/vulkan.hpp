#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE  // Use Vulkan depth range [0,1] instead of OpenGL [-1,1]

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
// stb_image: implementation is compiled in one translation unit (VulkanApp.cpp). Do not include the header here to avoid implementation duplication. // use <stb/stb_image.h> in .cpp files that need it.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp> 
#include <glm/matrix.hpp>

#include "../utils/FileReader.hpp"

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

// Allow forcing validation layers on in release/prod builds via the
// VK_VALIDATION environment variable. This is essential for hunting
// down GPU hangs/deadlocks that only reproduce in optimized builds
// (where validation is otherwise off). Sync validation is enabled
// automatically whenever validation layers are on (see createInstance).
inline bool validationLayersRequested() {
    if (const char* v = std::getenv("VK_VALIDATION")) {
        const char c = v[0];
        if (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T') return true;
    }
    return false;
}

#ifdef NDEBUG
const bool enableValidationLayers = validationLayersRequested();
#else
const bool enableValidationLayers = true;
#endif

// Forward declarations for types defined in implementation headers
class VulkanApp;
struct MaterialProperties;

// Vertex definition from math
#include "../math/Vertex.hpp"


// Small helper to create shader stage infos
#include "ShaderStage.hpp"


// Core Vulkan helper types used across the module
#include "Buffer.hpp"
#include "TextureImage.hpp"
#include "QueueFamilyIndices.hpp"
#include "VertexBufferObject.hpp"


// Note: Do not include "VulkanApp.hpp" here to avoid circular includes.
// Translation units that need the full `VulkanApp` definition should
// include "VulkanApp.hpp" explicitly.