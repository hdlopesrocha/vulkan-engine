#pragma once
#include "Widget.hpp"
#include "../Uniforms.hpp"
#include <vulkan/vulkan.h>
#include "../vulkan/Buffer.hpp"
#include "../vulkan/CubeToEquirectRenderer.hpp"

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

    // ImGui texture descriptors
    VkDescriptorSet skyDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solid360Descriptor = VK_NULL_HANDLE;
    VkDescriptorSet cube360EquirectDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet cube360FaceDescriptor[6] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet solidColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet backFaceDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterDepthLinearDescriptor = VK_NULL_HANDLE;

    CubeToEquirectRenderer cube360EquirectRenderer;

    // Ownership flags: true if this widget created the descriptor via ImGui_ImplVulkan_AddTexture
    bool skyDescriptorOwned = false;
    bool solid360DescriptorOwned = false;
    bool cube360EquirectDescriptorOwned = false;
    bool cube360FaceDescriptorOwned[6] = { false, false, false, false, false, false };
    bool solidColorDescriptorOwned = false;
    int selectedCubeFaceIndex = 0;
    bool solidDepthDescriptorOwned = false;
    bool waterColorDescriptorOwned = false;
    bool backFaceDepthDescriptorOwned = false;
    bool waterDepthLinearDescriptorOwned = false;

    // Converted linear depth debug images (device local) and views
    VkImage linearSceneDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory linearSceneDepthMemory = VK_NULL_HANDLE;
    VkImageView linearSceneDepthView = VK_NULL_HANDLE;
    VkDescriptorSet linearSceneDepthDescriptor = VK_NULL_HANDLE;
    bool linearSceneDepthDescriptorOwned = false;

    VkImage linearBackFaceDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory linearBackFaceDepthMemory = VK_NULL_HANDLE;
    VkImageView linearBackFaceDepthView = VK_NULL_HANDLE;
    VkDescriptorSet linearBackFaceDepthDescriptor = VK_NULL_HANDLE;
    bool linearBackFaceDepthDescriptorOwned = false;

    // GPU linearization pass resources
    VkRenderPass linearizeRenderPass = VK_NULL_HANDLE;
    VkPipeline linearizePipeline = VK_NULL_HANDLE;
    VkPipelineLayout linearizePipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout linearizeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet linearizeDescriptorSet = VK_NULL_HANDLE;
    VkFramebuffer linearSceneFramebuffer = VK_NULL_HANDLE;
    VkFramebuffer linearBackFaceFramebuffer = VK_NULL_HANDLE;

    // Single preview descriptor (widget displays one texture at a time)
    VkDescriptorSet previewDescriptor = VK_NULL_HANDLE;

    // Per-cascade linearized shadow debug images
    VkImage linearShadowDepthImage[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkDeviceMemory linearShadowDepthMemory[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkImageView linearShadowDepthView[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    VkDescriptorSet linearShadowDepthDescriptor[SHADOW_CASCADE_COUNT] = { VK_NULL_HANDLE };
    bool linearShadowDepthDescriptorOwned[SHADOW_CASCADE_COUNT] = { false };

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
        Solid360Equirect,
        SolidColor,
        SolidDepth,
        WaterWorldPos,
        WaterLinearDepth,
        WaterBackFace,
        LinearSceneDepth,
        LinearBackFaceDepth,
        ShadowCascade,
        Count
    };
    PreviewTarget selectedPreview = PreviewTarget::SolidColor;
    int selectedShadowCascade = 0;
    bool showAllCascades = false;
    enum class ShadowViewMode { Raw = 0, Linearized = 1 } shadowViewMode = ShadowViewMode::Linearized;

public:
    RenderTargetsWidget(VulkanApp* app, SceneRenderer* scene, SolidRenderer* solid, SkyRenderer* sky,
                        ShadowRenderer* shadow = nullptr, ShadowParams* shadowParams = nullptr, Settings* settings = nullptr);
    ~RenderTargetsWidget();

    void setFrameInfo(uint32_t frameIndex, int width, int height);
    // Destroy and recreate linear preview targets when size changes
    void destroyLinearTargets();
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
};
