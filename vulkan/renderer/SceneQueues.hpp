#pragma once

#include <vulkan/vulkan.h>
#include <vector>

// Scene/renderer-specific graphics queues.
//
// These encode *how* this particular engine stages its rendering passes:
// vegetation GPU culling, SDF, bounding-box, solid, water, sky, brush solid,
// brush liquid, and geometry compute. They are intentionally owned by the
// application (MyApp) and configured from VulkanApp's generic parallel
// graphics-queue pool, so the VulkanApp framework stays agnostic about *what*
// is rendered and *how* the scene is split across queues.
class SceneQueues {
public:
    VkQueue vegetationQueue = VK_NULL_HANDLE;
    VkQueue sdfQueue        = VK_NULL_HANDLE;
    VkQueue bboxQueue       = VK_NULL_HANDLE;
    VkQueue geometryQueue   = VK_NULL_HANDLE;
    VkQueue brushSolidQueue = VK_NULL_HANDLE;
    VkQueue brushLiquidQueue= VK_NULL_HANDLE;
    VkQueue solidQueue      = VK_NULL_HANDLE;
    VkQueue waterQueue      = VK_NULL_HANDLE;
    VkQueue skyQueue        = VK_NULL_HANDLE;

    VkQueue getVegetationQueue() const { return vegetationQueue; }
    VkQueue getSdfQueue() const { return sdfQueue; }
    VkQueue getBoundingBoxQueue() const { return bboxQueue; }
    VkQueue getGeometryQueue() const { return geometryQueue; }
    VkQueue getBrushSolidQueue() const { return brushSolidQueue; }
    VkQueue getBrushLiquidQueue() const { return brushLiquidQueue; }
    VkQueue getSolidQueue() const { return solidQueue; }
    VkQueue getWaterQueue() const { return waterQueue; }
    VkQueue getSkyQueue() const { return skyQueue; }

    // Assign each role from the device's parallel graphics-queue pool (`pg`),
    // which VulkanApp::createLogicalDevice builds from every acquired
    // graphics-family queue (deduplicated by handle, so aliased queues collapse
    // to one entry). When the device exposes fewer queues than a role needs, the
    // role aliases the single graphicsQueue — still correct, just no HW
    // parallelism. pg[1..9] correspond to acquireGfx(1..9).
    void configure(const std::vector<VkQueue>& pg, VkQueue graphicsQueue) {
        vegetationQueue   = (pg.size() > 1) ? pg[1] : graphicsQueue;
        sdfQueue          = (pg.size() > 2) ? pg[2] : graphicsQueue;
        bboxQueue         = (pg.size() > 3) ? pg[3] : graphicsQueue;
        geometryQueue     = (pg.size() > 4) ? pg[4] : graphicsQueue;
        solidQueue        = (pg.size() > 5) ? pg[5] : graphicsQueue;
        waterQueue        = (pg.size() > 6) ? pg[6] : graphicsQueue;
        skyQueue          = (pg.size() > 7) ? pg[7] : graphicsQueue;
        brushSolidQueue   = (pg.size() > 8) ? pg[8] : graphicsQueue;
        brushLiquidQueue  = (pg.size() > 9) ? pg[9] : graphicsQueue;
    }
};
