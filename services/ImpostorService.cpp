#include "ImpostorService.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"
#include <algorithm>

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
    srcAlbedo      = albedo;
    srcNormal      = normal;
    srcOpacity     = opacity;
    srcSampler     = sampler;

    if (vulkanApp && srcAlbedo != VK_NULL_HANDLE && srcSampler != VK_NULL_HANDLE) {
        captureAll(10.0f);
    }
}

void ImpostorService::captureAll(float scale) {
    if (!vulkanApp || srcAlbedo == VK_NULL_HANDLE || srcSampler == VK_NULL_HANDLE)
        return;

    // Snapshots bake at setup time, before any frame draw has synced the
    // UBOs. Push live fire settings now or fire captures bake stale init
    // defaults (tiny flames) inside live-sized frames. Wind stays unsynced
    // on purpose (calm, deterministic snapshots).
    if (vegRenderer) vegRenderer->updateFireParamsUBO();
    capture.captureAll(vulkanApp,
                       srcAlbedo, srcNormal, srcOpacity, srcSampler,
                       scale);
    rewire();
}

void ImpostorService::captureFireOnly() {
    if (!vulkanApp || !vegRenderer ||
        srcAlbedo == VK_NULL_HANDLE || srcSampler == VK_NULL_HANDLE)
        return;

    vegRenderer->updateFireParamsUBO();
    const float fireScale = std::max(0.1f, vegRenderer->getFireSettings().size);
    capture.capture(vulkanApp,
                    srcAlbedo, srcNormal, srcOpacity, srcSampler,
                    fireScale, VegetationRenderer::kFireBillboardIndex);
    // No rewire needed: the same array views already point at layers 60-79.
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
