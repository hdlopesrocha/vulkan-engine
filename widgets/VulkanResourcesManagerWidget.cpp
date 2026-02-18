#include "VulkanResourcesManagerWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <imgui.h>
#include <sstream>
#include <iomanip>

VulkanResourcesManagerWidget::VulkanResourcesManagerWidget(VulkanResourceManager* mgr, VulkanApp* app)
    : Widget("Vulkan Resources"), mgr(mgr), app(app) {}

static std::string handleToString(uint64_t v, bool hex) {
    std::ostringstream ss;
    if (hex) ss << "0x" << std::hex << v;
    else ss << std::dec << v;
    return ss.str();
}

void VulkanResourcesManagerWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    if (!mgr) {
        ImGui::TextUnformatted("No VulkanResourceManager reference");
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Show handles in hex", &showHex);
    ImGui::SameLine();
    static char filterBuf[128] = "";
    ImGui::InputTextWithHint("##filter", "filter (name or hex)", filterBuf, sizeof(filterBuf));
    ImGui::Separator();

    // Device-dependent helper
    VkDevice device = VK_NULL_HANDLE;
    if (app) device = app->getDevice();

    // App-level handles and swapchain info (if available)
    if (app) {
        if (ImGui::TreeNode("App / Instance")) {
            ImGui::Text("Instance: %s", handleToString(reinterpret_cast<uint64_t>(app->getInstance()), showHex).c_str());
            ImGui::Text("PhysicalDevice: %s", handleToString(reinterpret_cast<uint64_t>(app->getPhysicalDevice()), showHex).c_str());
            ImGui::Text("Device: %s", handleToString(reinterpret_cast<uint64_t>(app->getDevice()), showHex).c_str());
            ImGui::Text("GraphicsQueue: %s", handleToString(reinterpret_cast<uint64_t>(app->getGraphicsQueue()), showHex).c_str());
            ImGui::Text("PresentQueue: %s", handleToString(reinterpret_cast<uint64_t>(app->getPresentQueue()), showHex).c_str());

            ImGui::Separator();
            ImGui::Text("Swapchain: %s", handleToString(reinterpret_cast<uint64_t>(app->getSwapchain()), showHex).c_str());
            ImGui::Text("Swapchain format: %d  extent=%dx%d", (int)app->getSwapchainImageFormat(), app->getSwapchainExtent().width, app->getSwapchainExtent().height);
            ImGui::Text("Swapchain images: %zu", app->getSwapchainImages().size());
            ImGui::Text("DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(app->getDescriptorPool()), showHex).c_str());
            ImGui::Text("ImGui DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(app->getImGuiDescriptorPool()), showHex).c_str());

            ImGui::TreePop();
        }

        // Registered descriptor sets and pipelines
        const auto &rds = app->getRegisteredDescriptorSets();
        if (ImGui::TreeNode("Registered Descriptor Sets")) {
            if (rds.empty()) ImGui::TextUnformatted("(none registered)");
            for (size_t i = 0; i < rds.size(); ++i) {
                ImGui::Bullet(); ImGui::SameLine();
                ImGui::Text("Set %zu: %s", i, handleToString(reinterpret_cast<uint64_t>(rds[i]), showHex).c_str());
            }
            ImGui::TreePop();
        }

        const auto &rpips = app->getRegisteredPipelines();
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
        const auto &m = mgr->getDeviceMemoryMap();
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
            if (device != VK_NULL_HANDLE && app && app->isResourceRegistered((uintptr_t)handle)) {
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
            if (device != VK_NULL_HANDLE && app && app->isResourceRegistered((uintptr_t)handle)) {
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

    ImGui::End();
}
