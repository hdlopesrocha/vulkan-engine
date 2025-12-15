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
// stb_image for texture loading
#include <stb/stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp> 
#include <glm/matrix.hpp>

#include "FileReader.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#ifdef NDEBUG
const bool enableValidationLayers = false;
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