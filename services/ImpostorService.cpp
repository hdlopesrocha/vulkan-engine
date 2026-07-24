#include "ImpostorService.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/renderer/VegetationRenderer.hpp"

ImpostorService::ImpostorService() {}

void ImpostorService::init(VulkanApp* app) {
    vulkanApp = app;
    capture.init(app);
}

void ImpostorService::cleanup() {
    capture.cleanup(vulkanApp);
    srcAlbedo  = VK_NULL_HANDLE;
    srcNormal  = VK_NULL_HANDLE;
    srcOpacity = VK_NULL_HANDLE;
    srcSampler = VK_NULL_HANDLE;
    billboardCount = 0;
}

void ImpostorService::setSource(VkImageView albedo, VkImageView normal,
                                 VkImageView opacity, VkSampler sampler,
                                 int numBillboards) {
    srcAlbedo      = albedo;
    srcNormal      = normal;
    srcOpacity     = opacity;
    srcSampler     = sampler;
    billboardCount = numBillboards;

    if (vulkanApp && srcAlbedo != VK_NULL_HANDLE && srcSampler != VK_NULL_HANDLE) {
        captureAll(10.0f);
    }
}

void ImpostorService::captureAll(float scale) {
    if (!vulkanApp || srcAlbedo == VK_NULL_HANDLE || srcSampler == VK_NULL_HANDLE)
        return;

    capture.captureAll(vulkanApp,
                       srcAlbedo, srcNormal, srcOpacity, srcSampler,
                       scale);
    rewire();
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

ImTextureID ImpostorService::getImTextureID(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImTextureID(vulkanApp, billboardType, viewIdx);
}

ImTextureID ImpostorService::getImGuiNormalTextureID(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImGuiNormalTextureID(vulkanApp, billboardType, viewIdx);
}

ImTextureID ImpostorService::getImGuiDepthTextureID(uint32_t billboardType, uint32_t viewIdx) const {
    return capture.getImGuiDepthTextureID(vulkanApp, billboardType, viewIdx);
}
