#include "RendererUtils.hpp"

#include <cstdlib>
#include <iostream>

namespace RendererUtils {

void BarrierStats::endFrameReport(uint64_t warnThreshold) {
    static bool enabled = (std::getenv("VULKAN_BARRIER_STATS") != nullptr);
    if (!enabled) return;
    uint64_t idx = frameIndex.fetch_add(1, std::memory_order_relaxed);
    uint64_t calls = frameCalls.load(std::memory_order_relaxed);
    uint64_t imgBarriers = frameImageBarriers.load(std::memory_order_relaxed);
    std::cerr << "[BarrierStats] frame=" << idx
              << " vkCmdPipelineBarrier2 calls=" << calls
              << " imageBarriers=" << imgBarriers;
    if (calls > warnThreshold)
        std::cerr << " WARNING: exceeds target <=" << warnThreshold << "/frame";
    std::cerr << std::endl;
}

} // namespace RendererUtils
