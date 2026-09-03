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

uint32_t FrameGraph::addPass(const std::string& name) {
    FrameGraphPassNode node;
    node.name = name;
    passes_.push_back(std::move(node));
    return static_cast<uint32_t>(passes_.size() - 1);
}

void FrameGraph::addAccess(uint32_t pass, const FrameGraphResourceAccess& access) {
    if (pass >= passes_.size() || access.image == VK_NULL_HANDLE) return;
    passes_[pass].accesses.push_back(access);
}

std::vector<FrameGraph::MergedTransition> FrameGraph::buildMergedBarriers() const {
    // Group accesses by (image, baseLayer, layerCount) preserving first-seen
    // order. Nodes = passes in frame order, edges = shared resources: the
    // single transition per resource spans its first use to its last use.
    std::vector<MergedTransition> merged;
    auto findEntry = [&](VkImage img, uint32_t base, uint32_t count) -> MergedTransition* {
        for (auto& m : merged) {
            if (m.image == img && m.baseLayer == base && m.layerCount == count)
                return &m;
        }
        return nullptr;
    };
    for (const auto& pass : passes_) {
        for (const auto& acc : pass.accesses) {
            if (acc.image == VK_NULL_HANDLE) continue;
            MergedTransition* m = findEntry(acc.image, acc.baseLayer, acc.layerCount);
            if (!m) {
                MergedTransition e;
                e.image      = acc.image;
                e.format     = acc.format;
                e.oldLayout  = acc.layout;
                e.newLayout  = acc.layout;
                e.baseLayer  = acc.baseLayer;
                e.layerCount = acc.layerCount;
                e.isWrite    = acc.isWrite;
                merged.push_back(e);
            } else {
                // Extend the transition to the latest use in the frame.
                m->newLayout = acc.layout;
                m->isWrite   = m->isWrite || acc.isWrite;
                if (m->format == VK_FORMAT_UNDEFINED)
                    m->format = acc.format;
            }
        }
    }
    for (auto& m : merged)
        m.isNoOp = (m.oldLayout == m.newLayout && !m.isWrite);
    return merged;
}

uint32_t FrameGraph::emitMergedBarriers(VkCommandBuffer cmd, VulkanApp* app) const {
    if (cmd == VK_NULL_HANDLE || app == nullptr) return 0;
    std::vector<MergedTransition> merged = buildMergedBarriers();
    if (merged.empty()) return 0;

    // Translate into batch requests. No-op entries (layout unchanged,
    // read-only chain) are emitted with identical old/new layouts so the
    // batch layer records them as VK_ACCESS_2_NONE /
    // VK_PIPELINE_STAGE_2_NONE no-ops inside the SAME barrier call.
    std::vector<VulkanApp::BatchTransition> batch;
    batch.reserve(merged.size());
    for (const auto& m : merged) {
        VulkanApp::BatchTransition t;
        t.image          = m.image;
        t.format         = m.format;
        t.oldLayout      = m.oldLayout;
        t.newLayout      = m.newLayout;
        t.mipLevels      = 1;
        t.baseArrayLayer = m.baseLayer;
        t.layerCount     = m.layerCount;
        t.isNoOp         = m.isNoOp;
        batch.push_back(t);
    }
    return app->recordTransitionBatch(cmd, batch);
}

} // namespace RendererUtils
