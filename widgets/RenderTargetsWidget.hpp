#pragma once
#include "Widget.hpp"
#include "../vulkan/ubo/UniformObject.hpp"
#include <vulkan/vulkan.h>
#include "../vulkan/resources/Buffer.hpp"
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
class RenderTargetsWidget : public Widget {
private:
    VulkanApp*      app;
    SceneRenderer*  sceneRenderer;
    SolidRenderer*  solidRenderer;
    SkyRenderer*    skyRenderer;
    ShadowRenderer* shadowMapper;

    // ImGui texture descriptors
    VkDescriptorSet skyDescriptor = VK_NULL_HANDLE;
    // Hybrid RT output previews (reflection / refraction+thickness).
    VkDescriptorSet rtReflectDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet rtRefractDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet backFaceDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet brushBackFaceDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterDepthLinearDescriptor = VK_NULL_HANDLE;


    // Ownership flags: true if this widget created the descriptor via ImGui_ImplVulkan_AddTexture
    bool skyDescriptorOwned = false;
    bool rtReflectDescriptorOwned = false;
    bool rtRefractDescriptorOwned = false;
    bool solidColorDescriptorOwned = false;
    bool solidDepthDescriptorOwned = false;
    bool waterColorDescriptorOwned = false;
    bool backFaceDepthDescriptorOwned = false;
    bool waterDepthLinearDescriptorOwned = false;

    // Offscreen brush preview descriptors (SDF preview color + front depth)
    VkDescriptorSet brushColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet brushDepthDescriptor = VK_NULL_HANDLE;
    bool brushColorDescriptorOwned = false;
    bool brushDepthDescriptorOwned = false;

    // Offscreen debug previews: SDF debug cubes and mesh bounding boxes
    VkDescriptorSet sdfColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet sdfDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet bboxColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet bboxDepthDescriptor = VK_NULL_HANDLE;
    bool sdfColorDescriptorOwned = false;
    bool sdfDepthDescriptorOwned = false;
    bool bboxColorDescriptorOwned = false;
    bool bboxDepthDescriptorOwned = false;

    // Offscreen vegetation previews (color + depth)
    VkDescriptorSet vegColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet vegDepthDescriptor = VK_NULL_HANDLE;
    bool vegColorDescriptorOwned = false;
    bool vegDepthDescriptorOwned = false;

    // Converted linear depth debug images (device local) and views
    VkImage linearSceneDepthImage = VK_NULL_HANDLE;
    VmaAllocation linearSceneDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory linearSceneDepthMemory = VK_NULL_HANDLE;
    VkImageView linearSceneDepthView = VK_NULL_HANDLE;
    VkDescriptorSet linearSceneDepthDescriptor = VK_NULL_HANDLE;
    bool linearSceneDepthDescriptorOwned = false;

    VkImage waterDepthLinearImage = VK_NULL_HANDLE;
    VmaAllocation waterDepthLinearAllocation = VK_NULL_HANDLE;
    VkDeviceMemory waterDepthLinearMemory = VK_NULL_HANDLE;
    VkImageView waterDepthLinearView = VK_NULL_HANDLE;

    VkImage linearBackFaceDepthImage = VK_NULL_HANDLE;
    VmaAllocation linearBackFaceDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory linearBackFaceDepthMemory = VK_NULL_HANDLE;
    VkImageView linearBackFaceDepthView = VK_NULL_HANDLE;
    VkDescriptorSet linearBackFaceDepthDescriptor = VK_NULL_HANDLE;
    bool linearBackFaceDepthDescriptorOwned = false;

    // GPU linearization pass resources
    VkPipeline linearizePipeline = VK_NULL_HANDLE;
    VkPipelineLayout linearizePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet linearizeDescriptorSet = VK_NULL_HANDLE;
    // Widget sampler (widget no longer creates its own fallback sampler; it requires an app-provided sampler)
    VkSampler widgetSampler = VK_NULL_HANDLE;

    // Single preview descriptor (widget displays one texture at a time)
    VkDescriptorSet previewDescriptor = VK_NULL_HANDLE;

    // Per-cascade linearized shadow debug images
    VkImage linearShadowDepthImage[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VmaAllocation linearShadowDepthAllocation[SHADOW_CASCADE_COUNT] = {};
    VkDeviceMemory linearShadowDepthMemory[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkImageView linearShadowDepthView[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkDescriptorSet linearShadowDepthDescriptor[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    bool linearShadowDepthDescriptorOwned[SHADOW_CASCADE_COUNT] = { false };

    ShadowParams* shadowParams = nullptr;

    // Fixed preview width in pixels (all previews will be displayed at this width)
    // Preview scale removed — previews are fixed-size thumbnails now.
    int currentFrame = 0;
    int cachedWidth = 0;
    int cachedHeight = 0;

    // UI: which preview to show (only one at a time)
    enum class PreviewTarget {
        Sky = 0,
        RTReflect,          // hybrid RT water reflection output
        RTRefract,          // hybrid RT water refraction + thickness output
        SolidColor,
        SolidDepth,
        BackFaceColor,
        BackFaceDepth,
        BrushColor,
        BrushDepth,
        WaterColor,
        WaterDepth,
        SdfColor,
        SdfDepth,
        BboxColor,
        BboxDepth,
        VegColor,
        VegDepth,
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
    // Rely on renderer-provided tracked layouts.

public:
    RenderTargetsWidget(VulkanApp* app_, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                        ShadowRenderer* shadow = nullptr, ShadowParams* shadowParams_ = nullptr);
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
    // Called after ImGui is re-initialized (new DSL). Frees all owned ImGui AddTexture DS
    // (created with the old DSL) and resets them to VK_NULL_HANDLE so they are re-created
    // with the new DSL on the next frame.
    void invalidateImGuiDescriptors();
    // Run a small fullscreen pass that samples a depth image and writes a
    // normalized RGBA preview into `dstView`. `dstDescriptor` will be
    // created via ImGui_ImplVulkan_AddTexture if needed. `mode` selects
    // linearization mode: 0.0 = perspective linearize, 1.0 = passthrough.
    bool runLinearizePass(VulkanApp* app_, VkImage srcImage, VkImageView srcView, VkSampler srcSampler, VkSampler previewSampler,
                          VkImageView dstView,
                          VkDescriptorSet &dstDescriptor, bool &dstDescriptorOwned,
                          uint32_t width, uint32_t height,
                          float zNear, float zFar, float mode,
                          uint32_t srcBaseArrayLayer = 0);
};
