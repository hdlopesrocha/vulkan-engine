#include "VulkanResourcesManagerWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <imgui.h>
#include <sstream>
#include <iomanip>
#include "components/ImGuiHelpers.hpp"

VulkanResourcesManagerWidget::VulkanResourcesManagerWidget(VulkanResourceManager* mgr_)
    : Widget("Vulkan Resources", u8"\uf0e8"), mgr(mgr_) {}

void VulkanResourcesManagerWidget::updateWithApp(VulkanApp* app) {
    if (!app) { hasAppCache = false; return; }
    cachedInstance = app->getInstance();
    cachedPhysicalDevice = app->getPhysicalDevice();
    cachedDevice = app->getDevice();
    cachedGraphicsQueue = app->getGraphicsQueue();
    cachedPresentQueue = app->getPresentQueue();
    cachedSwapchain = app->getSwapchain();
    cachedSwapchainFormat = app->getSwapchainImageFormat();
    cachedSwapchainExtent = app->getSwapchainExtent();
    cachedRegisteredDescriptorSets = app->getRegisteredDescriptorSets();
    cachedRegisteredPipelines = app->getRegisteredPipelines();
    cachedDescriptorPool = app->getDescriptorPool();
    cachedImGuiDescriptorPool = app->getImGuiDescriptorPool();
    hasAppCache = true;

    // Cache logical queue handles and sample the activity history (one slot per
    // logical queue). The in-flight count is read from VulkanApp's per-queue
    // counters; it advances once per frame so the chart shows queue load over time.
    cachedQueue[Q_GRAPHICS]   = app->getGraphicsQueue();
    cachedQueue[Q_PRESENT]    = app->getPresentQueue();
    cachedQueue[Q_VEGETATION] = app->getVegetationQueue();
    cachedQueue[Q_SDF]        = app->getSdfQueue();
    cachedQueue[Q_BBOX]       = app->getBoundingBoxQueue();
    cachedQueue[Q_GEOMETRY]   = app->geometryTransferQueue();
    cachedQueue[Q_TRANSFER]   = app->getTransferQueue();

    static const char* names[Q_COUNT] = {
        "Graphics", "Present", "Vegetation", "SDF", "BoundingBox", "Geometry", "Transfer"
    };
    (void)names;

    for (int i = 0; i < Q_COUNT; ++i) {
        VkQueue q = cachedQueue[i];
        queueActive[i] = (q != VK_NULL_HANDLE);
        queuePendingNow[i]    = queueActive[i] ? app->getQueuePending(q)  : 0;
        queueSubmittedTotal[i]  = queueActive[i] ? app->getQueueSubmitted(q) : 0;
        queueCompletedTotal[i]  = queueActive[i] ? app->getQueueCompleted(q) : 0;

        // Shift history left and append the newest sample at the end.
        auto& h = queueHistory[i];
        for (int s = 0; s < QUEUE_HISTORY - 1; ++s) h[s] = h[s + 1];
        h[QUEUE_HISTORY - 1] = static_cast<float>(queuePendingNow[i]);
    }
}

static std::string handleToString(uint64_t v, bool hex) {
    std::ostringstream ss;
    if (hex) ss << "0x" << std::hex << v;
    else ss << std::dec << v;
    return ss.str();
}

void VulkanResourcesManagerWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    if (!mgr) {
        ImGui::TextUnformatted("No VulkanResourceManager reference");
        return;
    }

    ImGui::Checkbox("Show handles in hex", &showHex);
    ImGui::SameLine();
    static char filterBuf[128] = "";
    ImGui::InputTextWithHint("##filter", "filter (name or hex)", filterBuf, sizeof(filterBuf));
    ImGui::Separator();

    // Device-dependent helper
    VkDevice device = VK_NULL_HANDLE;
    if (hasAppCache) device = cachedDevice;

    // App-level handles and swapchain info (if available)
    if (hasAppCache) {
        if (ImGui::TreeNode("App / Instance")) {
            ImGui::Text("Instance: %s", handleToString(reinterpret_cast<uint64_t>(cachedInstance), showHex).c_str());
            ImGui::Text("PhysicalDevice: %s", handleToString(reinterpret_cast<uint64_t>(cachedPhysicalDevice), showHex).c_str());
            ImGui::Text("Device: %s", handleToString(reinterpret_cast<uint64_t>(cachedDevice), showHex).c_str());
            ImGui::Text("GraphicsQueue: %s", handleToString(reinterpret_cast<uint64_t>(cachedGraphicsQueue), showHex).c_str());
            ImGui::Text("PresentQueue: %s", handleToString(reinterpret_cast<uint64_t>(cachedPresentQueue), showHex).c_str());

            ImGui::Separator();
            ImGui::Text("Swapchain: %s", handleToString(reinterpret_cast<uint64_t>(cachedSwapchain), showHex).c_str());
            ImGui::Text("Swapchain format: %d  extent=%dx%d", (int)cachedSwapchainFormat, cachedSwapchainExtent.width, cachedSwapchainExtent.height);
            ImGui::Text("Swapchain images: %zu", cachedRegisteredDescriptorSets.size());
            ImGui::Text("DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(cachedDescriptorPool), showHex).c_str());
            ImGui::Text("ImGui DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(cachedImGuiDescriptorPool), showHex).c_str());

            ImGui::TreePop();
        }

        // Queue activity chart (one rolling line per logical queue). The y-axis is the
        // number of in-flight command buffers on that queue; a flat zero means idle,
        // a sustained high value means the queue is back-pressured (more work submitted
        // than the GPU completes per frame).
        if (ImGui::TreeNode("Queue Activity")) {
            static const char* qnames[Q_COUNT] = {
                "Graphics", "Present", "Vegetation", "SDF", "BoundingBox", "Geometry", "Transfer"
            };
            // Detect aliasing: group logical queues that share a VkQueue handle.
            for (int i = 0; i < Q_COUNT; ++i) {
                if (!queueActive[i]) continue;
                ImGui::Separator();
                ImGui::Text("%s queue", qnames[i]);
                // Alias annotation
                std::string aliases;
                for (int j = 0; j < Q_COUNT; ++j) {
                    if (j != i && queueActive[j] && cachedQueue[j] == cachedQueue[i])
                        aliases += std::string(aliases.empty() ? "" : ", ") + qnames[j];
                }
                if (!aliases.empty())
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "shares hardware queue with: %s", aliases.c_str());

                ImGui::Text("in-flight: %d   submitted: %llu   completed: %llu",
                    queuePendingNow[i],
                    (unsigned long long)queueSubmittedTotal[i],
                    (unsigned long long)queueCompletedTotal[i]);

                float vmin = 0.0f, vmax = 0.0f;
                for (int s = 0; s < QUEUE_HISTORY; ++s) { vmin = std::min(vmin, queueHistory[i][s]); vmax = std::max(vmax, queueHistory[i][s]); }
                if (vmax < 1.0f) vmax = 1.0f; // keep a stable 0..1 scale when idle
                char pltId[64];
                snprintf(pltId, sizeof(pltId), "##queuehist_%d", i);
                ImGui::PlotLines(pltId, queueHistory[i].data(), QUEUE_HISTORY, 0, nullptr, vmin, vmax, ImVec2(0.0f, 40.0f));
            }
            if (!hasAppCache)
                ImGui::TextUnformatted("(no app cache yet)");
            ImGui::TreePop();
        }

        // Registered descriptor sets and pipelines
        const auto &rds = cachedRegisteredDescriptorSets;
        if (ImGui::TreeNode("Registered Descriptor Sets")) {
            if (rds.empty()) ImGui::TextUnformatted("(none registered)");
            for (size_t i = 0; i < rds.size(); ++i) {
                ImGui::Bullet(); ImGui::SameLine();
                ImGui::Text("Set %zu: %s", i, handleToString(reinterpret_cast<uint64_t>(rds[i]), showHex).c_str());
            }
            ImGui::TreePop();
        }

        const auto &rpips = cachedRegisteredPipelines;
        if (ImGui::TreeNode("Registered Pipelines")) {
            if (rpips.empty()) ImGui::TextUnformatted("(none registered)");
            for (size_t i = 0; i < rpips.size(); ++i) {
                ImGui::Bullet(); ImGui::SameLine();
                ImGui::Text("Pipeline %zu: %s", i, handleToString(reinterpret_cast<uint64_t>(rpips[i]), showHex).c_str());
            }
            ImGui::TreePop();
        }
    }

    
    // Device memories
    if (ImGui::TreeNode("Device Memories")) {
        const auto m = mgr->getDeviceMemorySnapshot();
        size_t transientCount = 0;
        for (const auto &p : m) {
            const auto &desc = p.second.second;
            if (desc.rfind("Temp:", 0) == 0) ++transientCount;
        }
        ImGui::Text("Tracked: %zu (transient: %zu)", m.size(), transientCount);
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t i = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%zu: %s %s", i, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++i;
        }
        ImGui::TreePop();
    }

    // Images
    if (ImGui::TreeNode("Images")) {
        const auto &m = mgr->getImageMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            std::string label = "Image "; label += std::to_string(idx);
            const std::string handleStr = handleToString(static_cast<uint64_t>(handle), showHex);
            bool show = (filterBuf[0] == '\0') || (label.find(filterBuf) != std::string::npos) || (handleStr.find(filterBuf) != std::string::npos) || (desc.find(filterBuf) != std::string::npos);
            if (!show) { ++idx; continue; }
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%s: %s %s", label.c_str(), handleStr.c_str(), desc.c_str());
            if (device != VK_NULL_HANDLE && mgr->find(handle).has_value()) {
                VkMemoryRequirements mr = {};
                vkGetImageMemoryRequirements(device, reinterpret_cast<VkImage>(handle), &mr);
                ImGui::Text("    mem reqs: size=%llu alignment=%u memoryTypeBits=0x%X", (unsigned long long)mr.size, (unsigned)mr.alignment, mr.memoryTypeBits);
                VkImageSubresource sub = {};
                sub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                sub.mipLevel = 0;
                sub.arrayLayer = 0;
                VkSubresourceLayout layout = {};
                vkGetImageSubresourceLayout(device, reinterpret_cast<VkImage>(handle), &sub, &layout);
                ImGui::Text("    subresource layout: offset=%llu size=%llu rowPitch=%u arrayPitch=%u depthPitch=%u", (unsigned long long)layout.offset, (unsigned long long)layout.size, (unsigned)layout.rowPitch, (unsigned)layout.arrayPitch, (unsigned)layout.depthPitch);
            }
            ++idx;
        }
        ImGui::TreePop();
    }

    // Image views
    if (ImGui::TreeNode("Image Views")) {
        const auto &m = mgr->getImageViewMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            std::string label = "View "; label += std::to_string(idx);
            const std::string handleStr = handleToString(static_cast<uint64_t>(handle), showHex);
            if (filterBuf[0] != '\0' && label.find(filterBuf) == std::string::npos && handleStr.find(filterBuf) == std::string::npos && desc.find(filterBuf) == std::string::npos) { ++idx; continue; }
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%s: %s %s", label.c_str(), handleStr.c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Samplers
    if (ImGui::TreeNode("Samplers")) {
        const auto &m = mgr->getSamplerMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("Sampler %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Framebuffers
    if (ImGui::TreeNode("Framebuffers")) {
        const auto &m = mgr->getFramebufferMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("FB %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Buffers
    if (ImGui::TreeNode("Buffers")) {
        const auto &m = mgr->getBufferMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            std::string label = "Buffer "; label += std::to_string(idx);
            const std::string handleStr = handleToString(static_cast<uint64_t>(handle), showHex);
            if (filterBuf[0] != '\0' && label.find(filterBuf) == std::string::npos && handleStr.find(filterBuf) == std::string::npos && desc.find(filterBuf) == std::string::npos) { ++idx; continue; }
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%s: %s %s", label.c_str(), handleStr.c_str(), desc.c_str());
            if (device != VK_NULL_HANDLE && mgr->find(handle).has_value()) {
                VkMemoryRequirements mr = {};
                vkGetBufferMemoryRequirements(device, reinterpret_cast<VkBuffer>(handle), &mr);
                ImGui::Text("    mem reqs: size=%llu alignment=%u memoryTypeBits=0x%X", (unsigned long long)mr.size, (unsigned)mr.alignment, mr.memoryTypeBits);
            }
            ++idx;
        }
        ImGui::TreePop();
    }

    // Pipelines
    if (ImGui::TreeNode("Pipelines")) {
        const auto &m = mgr->getPipelineMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("Pipeline %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Pipeline layouts
    if (ImGui::TreeNode("Pipeline Layouts")) {
        const auto &m = mgr->getPipelineLayoutMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("Layout %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Shader modules
    if (ImGui::TreeNode("Shader Modules")) {
        const auto &m = mgr->getShaderModuleMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("Shader %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Descriptor pools
    if (ImGui::TreeNode("Descriptor Pools")) {
        const auto &m = mgr->getDescriptorPoolMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("Pool %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Descriptor sets
    if (ImGui::TreeNode("Descriptor Sets")) {
        const auto &m = mgr->getDescriptorSetMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            std::string label = "Descriptor Set "; label += std::to_string(idx);
            const std::string handleStr = handleToString(static_cast<uint64_t>(handle), showHex);
            if (filterBuf[0] != '\0' && label.find(filterBuf) == std::string::npos && handleStr.find(filterBuf) == std::string::npos && desc.find(filterBuf) == std::string::npos) { ++idx; continue; }
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%s: %s %s", label.c_str(), handleStr.c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }

    // Descriptor set layouts
    if (ImGui::TreeNode("Descriptor Set Layouts")) {
        const auto &m = mgr->getDescriptorSetLayoutMap();
        if (m.empty()) ImGui::TextUnformatted("(none)");
        size_t idx = 0;
        for (const auto &p : m) {
            auto handle = p.first;
            const auto &desc = p.second.second;
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("DSL %zu: %s %s", idx, handleToString(static_cast<uint64_t>(handle), showHex).c_str(), desc.c_str());
            ++idx;
        }
        ImGui::TreePop();
    }
}
