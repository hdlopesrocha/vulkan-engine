#pragma once
#include "Widget.hpp"
#include "../vulkan/ubo/UniformObject.hpp"
#include <imgui.h>
#include <vulkan/vulkan.h>
#include "../vulkan/Buffer.hpp"
#include "../vulkan/renderer/CubeToEquirectRenderer.hpp"
#include <unordered_map>

class VulkanApp;
class WaterRenderer;
class SceneRenderer;
class SolidRenderer;
class SkyRenderer;
class ShadowRenderer;
struct ShadowParams;

// Widget that displays render targets (Sky, Solid color/depth, Water depth,
// Shadow cascades) as ImGui image thumbnails.
class Settings;

class RenderTargetsWidget : public Widget {
private:
    VulkanApp*      app;
    SceneRenderer*  sceneRenderer;
    SolidRenderer*  solidRenderer;
    SkyRenderer*    skyRenderer;
    ShadowRenderer* shadowMapper;

    CubeToEquirectRenderer cube360EquirectRenderer;

    int selectedCubeFaceIndex = 0;

    // Converted linear depth debug images (device local) and views
    VkImage linearSceneDepthImage = VK_NULL_HANDLE;
    VmaAllocation linearSceneDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory linearSceneDepthMemory = VK_NULL_HANDLE;
    VkImageView linearSceneDepthView = VK_NULL_HANDLE;

    VkImage waterDepthLinearImage = VK_NULL_HANDLE;
    VmaAllocation waterDepthLinearAllocation = VK_NULL_HANDLE;
    VkDeviceMemory waterDepthLinearMemory = VK_NULL_HANDLE;
    VkImageView waterDepthLinearView = VK_NULL_HANDLE;

    VkImage linearBackFaceDepthImage = VK_NULL_HANDLE;
    VmaAllocation linearBackFaceDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory linearBackFaceDepthMemory = VK_NULL_HANDLE;
    VkImageView linearBackFaceDepthView = VK_NULL_HANDLE;

    VkImage linearBrushBackFaceDepthImage = VK_NULL_HANDLE;
    VmaAllocation linearBrushBackFaceDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory linearBrushBackFaceDepthMemory = VK_NULL_HANDLE;
    VkImageView linearBrushBackFaceDepthView = VK_NULL_HANDLE;

    // Per-face linearized targets for cubemap depth previews
    VkImage linearCubeFaceDepthImage[6] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation linearCubeFaceDepthAllocation[6] = {};
    VkDeviceMemory linearCubeFaceDepthMemory[6] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView linearCubeFaceDepthView[6] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkFramebuffer linearCubeFaceFramebuffer[6] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE }; // unused - dynamic rendering

    // GPU linearization pass resources
    VkPipeline linearizePipeline = VK_NULL_HANDLE;
    VkPipelineLayout linearizePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet linearizeDescriptorSet = VK_NULL_HANDLE;
    // Widget sampler (widget no longer creates its own fallback sampler; it requires an app-provided sampler)
    VkSampler widgetSampler = VK_NULL_HANDLE;

    // Single preview texture ID (widget displays one texture at a time).
    // Set each frame in updateDescriptors() via ImTextureManager.
    ImTextureID previewTextureID = 0;

    // Per-cascade linearized shadow debug images
    VkImage linearShadowDepthImage[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VmaAllocation linearShadowDepthAllocation[SHADOW_CASCADE_COUNT] = {};
    VkDeviceMemory linearShadowDepthMemory[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkImageView linearShadowDepthView[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };

    ShadowParams* shadowParams = nullptr;
    Settings* settings = nullptr; // pointer to app settings for near/far

    // Fixed preview width in pixels (all previews will be displayed at this width)
    // Preview scale removed — previews are fixed-size thumbnails now.
    int currentFrame = 0;
    int cachedWidth = 0;
    int cachedHeight = 0;
    // GPU-only preview (always enabled; no CPU readback)
    bool useGpuPreview = true;
    // Persistent staging buffers (kept for compatibility but unused)
    Buffer stagingReadBuffer{};
    void* stagingReadPtr = nullptr;
    VkDeviceSize stagingReadSize = 0;

    Buffer stagingUploadBuffer{}; // used as TRANSFER_SRC for buffer->image uploads
    void* stagingUploadPtr = nullptr;
    VkDeviceSize stagingUploadSize = 0;

    int frameCounter = 0;
    int updateInterval = 8; // update debug readbacks every N frames (reduce CPU roundtrips)
    // Track sizes of linear debug images to detect resizes
    int linearSceneWidth = 0;
    int linearSceneHeight = 0;
    int linearShadowSize[SHADOW_CASCADE_COUNT] = { 0 };

    // UI: which preview to show (only one at a time)
    enum class PreviewTarget {
        Sky = 0,
        Solid360Cube,
        Solid360DepthCube,
        Solid360Equirect,
        SolidColor,
        SolidDepth,
        BackFaceColor,
        BackFaceDepth,
        BrushColor,
        BrushDepth,
        BrushBackFaceDepth,
        WaterColor,
        WaterDepth,
        LinearSceneDepth,
        ShadowCascade,
        Count
    };
    PreviewTarget selectedPreview = PreviewTarget::SolidColor;
    int selectedShadowCascade = 0;
    bool showAllCascades = false;
    enum class ShadowViewMode { Raw = 0, Linearized = 1 } shadowViewMode = ShadowViewMode::Linearized;

    // Auto-advance: cycle through all preview targets every N frames
    bool autoAdvance = true;
    int autoAdvanceInterval = 10;  // frames between advances
    int autoAdvanceFrameCounter = 0;

    // NOTE: widget no longer maintains fallbacks or heuristic layout maps.
    // Rely on renderer-provided tracked layouts (e.g. Solid360Renderer).

public:
    RenderTargetsWidget(VulkanApp* app_, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                        ShadowRenderer* shadow = nullptr, ShadowParams* shadowParams_ = nullptr, Settings* settings_ = nullptr);
    ~RenderTargetsWidget();

    // Initialize static GPU resources used by the widget (run once).
    // If width/height are provided, create size-dependent targets too.
    void init(VulkanApp* app_, int width = 0, int height = 0);

    void setFrameInfo(uint32_t frameIndex, int width, int height);
    // Destroy and recreate linear preview targets when size changes
    void destroyLinearTargets();
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
    // True when the currently selected preview is one of the Solid360 cubemap
    // targets (so the cubemap must be kept up to date even when water is disabled).
    bool isSolid360Preview() const;
    // Run a small fullscreen pass that samples a depth image and writes a
    // normalized RGBA preview into `dstView`. `mode` selects
    // linearization mode: 0.0 = perspective linearize, 1.0 = passthrough.
    bool runLinearizePass(VulkanApp* app_, VkImage srcImage, VkImageView srcView, VkSampler srcSampler, VkSampler previewSampler,
                          VkImageView dstView,
                          uint32_t width, uint32_t height,
                          float zNear, float zFar, float mode,
                          uint32_t srcBaseArrayLayer = 0);
};
