#include "ImpostorService.hpp"
#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"

ImpostorService::ImpostorService() {}

void ImpostorService::init(VulkanApp* app) {
    vulkanApp = app;
    // The vegetation renderer (set via setVegetationRenderer before init)
    // supplies the shared set=2 wind params descriptor set/layout used by
    // the capture pipeline, instead of ImpostorCapture allocating a
    // duplicate wind params UBO + descriptor set.
    capture.init(app, vegRenderer);
}

void ImpostorService::cleanup() {
    capture.cleanup(vulkanApp);
    srcAlbedo  = VK_NULL_HANDLE;
    srcNormal  = VK_NULL_HANDLE;
    srcOpacity = VK_NULL_HANDLE;
    srcSampler = VK_NULL_HANDLE;
}

void ImpostorService::setSource(VkImageView albedo, VkImageView normal,
                                 VkImageView opacity, VkSampler sampler) {
    // Bladed grass capture is procedural: source views are ignored (kept for
    // caller compatibility). Capture immediately.
    (void)albedo; (void)normal; (void)opacity; (void)sampler;
    srcAlbedo = VK_NULL_HANDLE;
    srcNormal = VK_NULL_HANDLE;
    srcOpacity = VK_NULL_HANDLE;
    srcSampler = VK_NULL_HANDLE;

    if (vulkanApp) {
        captureAll(10.0f);
    }
}

void ImpostorService::captureAll(float scale) {
    if (!vulkanApp)
        return;

    // Async whole-array capture: one submit for all 60 views, completion
    // published through the deferred-fence queue (completeCaptureType). Setup
    // must not block on it — the vegetation renderer is rewired from the same
    // fence callback once the arrays are actually sampleable.
    VkFence fence = capture.captureAllAsync(vulkanApp,
                                            VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            VK_NULL_HANDLE, VK_NULL_HANDLE,
                                            scale);
    if (fence != VK_NULL_HANDLE) {
        // Deferred callbacks for one fence run in registration order, so this
        // is after completeCaptureType() marked the types captured.
        vulkanApp->deferDestroyUntilFence(fence, [this]() { rewire(); });
    }
}

void ImpostorService::rewire() {
    if (vulkanApp && vegRenderer && capture.isReady() &&
        capture.getCaptureArrayView() != VK_NULL_HANDLE) {
        vegRenderer->setImpostorData(vulkanApp,
                                     capture.getCaptureArrayView(),
                                     capture.getCaptureNormalArrayView(),
                                     capture.getCaptureArraySampler(),
                                     capture.getCaptureDepthArrayView(),
                                     capture.getCaptureInvVPBuffer());
    }
}

void ImpostorService::invalidateImGuiDescriptors() {
    capture.invalidateImGuiDescriptors();
}

void ImpostorService::recreateImGuiDescriptors() {
    if (vulkanApp && capture.isReady()) {
        capture.recreateAllImGuiDescSets(vulkanApp);
    }
}

VkDescriptorSet ImpostorService::getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImGuiDescSet(billboardType, viewIdx);
}

VkDescriptorSet ImpostorService::getImGuiNormalDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImGuiNormalDescSet(billboardType, viewIdx);
}

VkDescriptorSet ImpostorService::getImGuiDepthDescSet(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImGuiDepthDescSet(billboardType, viewIdx);
}
