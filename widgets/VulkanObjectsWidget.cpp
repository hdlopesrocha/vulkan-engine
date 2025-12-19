#include "VulkanObjectsWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <imgui.h>
#include <sstream>
#include <iomanip>

VulkanObjectsWidget::VulkanObjectsWidget(VulkanApp* app)
    : Widget("Vulkan Objects"), app(app) {}

static std::string handleToString(uint64_t v, bool hex) {
    std::ostringstream ss;
    if (hex) ss << "0x" << std::hex << v;
    else ss << std::dec << v;
    return ss.str();
}

void VulkanObjectsWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    if (!app) {
        ImGui::TextUnformatted("No VulkanApp reference");
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Show handles in hex", &showHex);
    ImGui::Separator();

    ImGui::Text("Instance: %s", handleToString(reinterpret_cast<uint64_t>(app->getInstance()), showHex).c_str());
    ImGui::Text("PhysicalDevice: %s", handleToString(reinterpret_cast<uint64_t>(app->getPhysicalDevice()), showHex).c_str());
    ImGui::Text("Device: %s", handleToString(reinterpret_cast<uint64_t>(app->getDevice()), showHex).c_str());
    ImGui::Text("GraphicsQueue: %s", handleToString(reinterpret_cast<uint64_t>(app->getGraphicsQueue()), showHex).c_str());
    ImGui::Text("PresentQueue: %s", handleToString(reinterpret_cast<uint64_t>(app->getPresentQueue()), showHex).c_str());
    ImGui::Separator();

    ImGui::Text("Swapchain: %s", handleToString(reinterpret_cast<uint64_t>(app->getSwapchain()), showHex).c_str());
    ImGui::Text("Swapchain format: %d extent=%dx%d", (int)app->getSwapchainImageFormat(), app->getSwapchainExtent().width, app->getSwapchainExtent().height);
    const auto& imgs = app->getSwapchainImages();
    const auto& ivs = app->getSwapchainImageViews();
    if (ImGui::TreeNode("Swapchain Images")) {
        for (size_t i = 0; i < imgs.size(); ++i) {
            std::string label = "Image "; label += std::to_string(i);
            ImGui::Bullet(); ImGui::SameLine();
            ImGui::Text("%s: %s", label.c_str(), handleToString(reinterpret_cast<uint64_t>(imgs[i]), showHex).c_str());
            if (i < ivs.size()) {
                ImGui::Text("    view: %s", handleToString(reinterpret_cast<uint64_t>(ivs[i]), showHex).c_str());
            }
        }
        ImGui::TreePop();
    }

    ImGui::Separator();
    ImGui::Text("DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(app->getDescriptorPool()), showHex).c_str());
    ImGui::Text("ImGui DescriptorPool: %s", handleToString(reinterpret_cast<uint64_t>(app->getImGuiDescriptorPool()), showHex).c_str());

    const auto &ds = app->getRegisteredDescriptorSets();
    if (ImGui::TreeNode("Descriptor Sets")) {
        if (ds.empty()) {
            ImGui::TextUnformatted("(none registered)");
        } else {
            for (size_t i = 0; i < ds.size(); ++i) {
                ImGui::Bullet(); ImGui::SameLine();
                std::string label = "Set "; label += std::to_string(i);
                ImGui::Text("%s: %s", label.c_str(), handleToString(reinterpret_cast<uint64_t>(ds[i]), showHex).c_str());
            }
        }
        ImGui::TreePop();
    }

    const auto &pips = app->getRegisteredPipelines();
    if (ImGui::TreeNode("Graphics Pipelines")) {
        if (pips.empty()) {
            ImGui::TextUnformatted("(none registered)");
        } else {
            ImGui::Text("Count: %zu", pips.size());
            for (size_t i = 0; i < pips.size(); ++i) {
                ImGui::Bullet(); ImGui::SameLine();
                std::string label = "Pipeline "; label += std::to_string(i);
                ImGui::Text("%s: %s", label.c_str(), handleToString(reinterpret_cast<uint64_t>(pips[i]), showHex).c_str());
            }
        }
        ImGui::TreePop();
    }

    ImGui::End();
}
