// Single authoritative implementation of VulkanResourceManager
#include "VulkanResourceManager.hpp"

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

using namespace std;

void VulkanResourceManager::addDeviceMemory(VkDeviceMemory mem, const char* desc) {
    if (mem == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return; // ImGui-managed resources should not be tracked here
    }
    std::lock_guard<std::mutex> lk(mtx);
    deviceMemories[(uintptr_t)mem] = {mem, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addImage(VkImage img, const char* desc) {
    if (img == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    images[(uintptr_t)img] = {img, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addImageView(VkImageView iv, const char* desc) {
    if (iv == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    imageViews[(uintptr_t)iv] = {iv, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addSampler(VkSampler s, const char* desc) {
    if (s == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    samplers[(uintptr_t)s] = {s, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addFramebuffer(VkFramebuffer fb, const char* desc) {
    if (fb == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    framebuffers[(uintptr_t)fb] = {fb, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addBuffer(VkBuffer b, const char* desc) {
    if (b == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    buffers[(uintptr_t)b] = {b, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addPipeline(VkPipeline p, const char* desc) {
    if (p == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    pipelines[(uintptr_t)p] = {p, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addPipelineLayout(VkPipelineLayout pl, const char* desc) {
    if (pl == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    pipelineLayouts[(uintptr_t)pl] = {pl, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addShaderModule(VkShaderModule m, const char* desc) {
    if (m == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    shaderModules[(uintptr_t)m] = {m, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addDescriptorPool(VkDescriptorPool dp, const char* desc) {
    if (dp == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    descriptorPools[(uintptr_t)dp] = {dp, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addDescriptorSet(VkDescriptorSet ds, const char* desc) {
    if (ds == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    descriptorSets[(uintptr_t)ds] = {ds, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addDescriptorSetLayout(VkDescriptorSetLayout dsl, const char* desc) {
    if (dsl == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    descriptorSetLayouts[(uintptr_t)dsl] = {dsl, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addRenderPass(VkRenderPass rp, const char* desc) {
    if (rp == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    renderPasses[(uintptr_t)rp] = {rp, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addSemaphore(VkSemaphore s, const char* desc) {
    if (s == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    semaphores[(uintptr_t)s] = {s, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addFence(VkFence f, const char* desc) {
    if (f == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    fences[(uintptr_t)f] = {f, desc ? std::string(desc) : std::string()};
}

void VulkanResourceManager::addCommandPool(VkCommandPool cp, const char* desc) {
    if (cp == VK_NULL_HANDLE) return;
    if (desc) {
        std::string d(desc);
        if (d.find("ImGui:") != std::string::npos) return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    commandPools[(uintptr_t)cp] = {cp, desc ? std::string(desc) : std::string()};
}

// Remove methods
void VulkanResourceManager::removeDeviceMemory(VkDeviceMemory mem) { std::lock_guard<std::mutex> lk(mtx); deviceMemories.erase((uintptr_t)mem); }
void VulkanResourceManager::removeImage(VkImage img) { std::lock_guard<std::mutex> lk(mtx); images.erase((uintptr_t)img); }
void VulkanResourceManager::removeImageView(VkImageView iv) { std::lock_guard<std::mutex> lk(mtx); imageViews.erase((uintptr_t)iv); }
void VulkanResourceManager::removeSampler(VkSampler s) { std::lock_guard<std::mutex> lk(mtx); samplers.erase((uintptr_t)s); }
void VulkanResourceManager::removeFramebuffer(VkFramebuffer fb) { std::lock_guard<std::mutex> lk(mtx); framebuffers.erase((uintptr_t)fb); }
void VulkanResourceManager::removeBuffer(VkBuffer b) { std::lock_guard<std::mutex> lk(mtx); buffers.erase((uintptr_t)b); }
void VulkanResourceManager::removePipeline(VkPipeline p) { std::lock_guard<std::mutex> lk(mtx); pipelines.erase((uintptr_t)p); }
void VulkanResourceManager::removePipelineLayout(VkPipelineLayout pl) { std::lock_guard<std::mutex> lk(mtx); pipelineLayouts.erase((uintptr_t)pl); }
void VulkanResourceManager::removeShaderModule(VkShaderModule m) { std::lock_guard<std::mutex> lk(mtx); shaderModules.erase((uintptr_t)m); }
void VulkanResourceManager::removeDescriptorPool(VkDescriptorPool dp) { std::lock_guard<std::mutex> lk(mtx); descriptorPools.erase((uintptr_t)dp); }
void VulkanResourceManager::removeDescriptorSet(VkDescriptorSet ds) { std::lock_guard<std::mutex> lk(mtx); descriptorSets.erase((uintptr_t)ds); }
void VulkanResourceManager::removeDescriptorSetLayout(VkDescriptorSetLayout dsl) { std::lock_guard<std::mutex> lk(mtx); descriptorSetLayouts.erase((uintptr_t)dsl); }
void VulkanResourceManager::removeRenderPass(VkRenderPass rp) { std::lock_guard<std::mutex> lk(mtx); renderPasses.erase((uintptr_t)rp); }
void VulkanResourceManager::removeSemaphore(VkSemaphore s) { std::lock_guard<std::mutex> lk(mtx); semaphores.erase((uintptr_t)s); }
void VulkanResourceManager::removeFence(VkFence f) { std::lock_guard<std::mutex> lk(mtx); fences.erase((uintptr_t)f); }
void VulkanResourceManager::removeCommandPool(VkCommandPool cp) { std::lock_guard<std::mutex> lk(mtx); commandPools.erase((uintptr_t)cp); }

// Accessors
const VulkanResourceManager::ResourceMap<VkDeviceMemory> &VulkanResourceManager::getDeviceMemoryMap() const { return deviceMemories; }
const VulkanResourceManager::ResourceMap<VkImage> &VulkanResourceManager::getImageMap() const { return images; }
const VulkanResourceManager::ResourceMap<VkImageView> &VulkanResourceManager::getImageViewMap() const { return imageViews; }
const VulkanResourceManager::ResourceMap<VkSampler> &VulkanResourceManager::getSamplerMap() const { return samplers; }
const VulkanResourceManager::ResourceMap<VkFramebuffer> &VulkanResourceManager::getFramebufferMap() const { return framebuffers; }
const VulkanResourceManager::ResourceMap<VkBuffer> &VulkanResourceManager::getBufferMap() const { return buffers; }
const VulkanResourceManager::ResourceMap<VkPipeline> &VulkanResourceManager::getPipelineMap() const { return pipelines; }
const VulkanResourceManager::ResourceMap<VkPipelineLayout> &VulkanResourceManager::getPipelineLayoutMap() const { return pipelineLayouts; }
const VulkanResourceManager::ResourceMap<VkShaderModule> &VulkanResourceManager::getShaderModuleMap() const { return shaderModules; }
const VulkanResourceManager::ResourceMap<VkDescriptorPool> &VulkanResourceManager::getDescriptorPoolMap() const { return descriptorPools; }
const VulkanResourceManager::ResourceMap<VkDescriptorSet> &VulkanResourceManager::getDescriptorSetMap() const { return descriptorSets; }
const VulkanResourceManager::ResourceMap<VkDescriptorSetLayout> &VulkanResourceManager::getDescriptorSetLayoutMap() const { return descriptorSetLayouts; }
const VulkanResourceManager::ResourceMap<VkRenderPass> &VulkanResourceManager::getRenderPassMap() const { return renderPasses; }
const VulkanResourceManager::ResourceMap<VkSemaphore> &VulkanResourceManager::getSemaphoreMap() const { return semaphores; }
const VulkanResourceManager::ResourceMap<VkFence> &VulkanResourceManager::getFenceMap() const { return fences; }
const VulkanResourceManager::ResourceMap<VkCommandPool> &VulkanResourceManager::getCommandPoolMap() const { return commandPools; }




// Find by handle
std::optional<VulkanResourceManager::Entry> VulkanResourceManager::find(uintptr_t handle) const {
    std::lock_guard<std::mutex> lk(mtx);
    auto check = [&](const auto &m, VkObjectType t) -> std::optional<Entry> {
        auto it2 = m.find(handle);
        if (it2 != m.end()) return Entry{t, it2->second.second};
        return std::nullopt;
    };

    if (auto e = check(deviceMemories, VK_OBJECT_TYPE_DEVICE_MEMORY)) return e;
    if (auto e = check(images, VK_OBJECT_TYPE_IMAGE)) return e;
    if (auto e = check(imageViews, VK_OBJECT_TYPE_IMAGE_VIEW)) return e;
    if (auto e = check(samplers, VK_OBJECT_TYPE_SAMPLER)) return e;
    if (auto e = check(framebuffers, VK_OBJECT_TYPE_FRAMEBUFFER)) return e;
    if (auto e = check(buffers, VK_OBJECT_TYPE_BUFFER)) return e;
    if (auto e = check(pipelines, VK_OBJECT_TYPE_PIPELINE)) return e;
    if (auto e = check(pipelineLayouts, VK_OBJECT_TYPE_PIPELINE_LAYOUT)) return e;
    if (auto e = check(shaderModules, VK_OBJECT_TYPE_SHADER_MODULE)) return e;
    if (auto e = check(descriptorPools, VK_OBJECT_TYPE_DESCRIPTOR_POOL)) return e;
    if (auto e = check(descriptorSets, VK_OBJECT_TYPE_DESCRIPTOR_SET)) return e;
    if (auto e = check(descriptorSetLayouts, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)) return e;
    if (auto e = check(renderPasses, VK_OBJECT_TYPE_RENDER_PASS)) return e;
    if (auto e = check(semaphores, VK_OBJECT_TYPE_SEMAPHORE)) return e;
    if (auto e = check(fences, VK_OBJECT_TYPE_FENCE)) return e;
    if (auto e = check(commandPools, VK_OBJECT_TYPE_COMMAND_POOL)) return e;

    return std::nullopt;
}



// Cleanup in a safe deterministic order
void VulkanResourceManager::cleanup(VkDevice device) {
    if (device == VK_NULL_HANDLE) return;

    // Helper destroys by extracting keys while holding mutex only for map access/erase
    auto destroyAndClear = [&](auto &m, auto destroyFn) {
        std::vector<uintptr_t> keys;
        {
            std::lock_guard<std::mutex> lk(mtx);
            keys.reserve(m.size());
            for (const auto &p : m) keys.push_back(p.first);
        }
        for (uintptr_t h : keys) {
            destroyFn(device, h);
            std::lock_guard<std::mutex> lk(mtx);
            m.erase(h);
        }
    };

    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying pipelines\n"); fflush(stderr);
    destroyAndClear(pipelines, [](VkDevice d, uintptr_t h){ vkDestroyPipeline(d, reinterpret_cast<VkPipeline>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying pipeline layouts\n"); fflush(stderr);
    destroyAndClear(pipelineLayouts, [](VkDevice d, uintptr_t h){ vkDestroyPipelineLayout(d, reinterpret_cast<VkPipelineLayout>(h), nullptr); });
    // Destroy shader modules explicitly to satisfy validation layers.
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying shaderModules\n"); fflush(stderr);
    destroyAndClear(shaderModules, [](VkDevice d, uintptr_t h){ vkDestroyShaderModule(d, reinterpret_cast<VkShaderModule>(h), nullptr); });
    
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying framebuffers\n"); fflush(stderr);
    destroyAndClear(framebuffers, [](VkDevice d, uintptr_t h){ vkDestroyFramebuffer(d, reinterpret_cast<VkFramebuffer>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying samplers\n"); fflush(stderr);
    destroyAndClear(samplers, [](VkDevice d, uintptr_t h){ vkDestroySampler(d, reinterpret_cast<VkSampler>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying imageViews\n"); fflush(stderr);
    destroyAndClear(imageViews, [](VkDevice d, uintptr_t h){ vkDestroyImageView(d, reinterpret_cast<VkImageView>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying images\n"); fflush(stderr);
    destroyAndClear(images, [](VkDevice d, uintptr_t h){ vkDestroyImage(d, reinterpret_cast<VkImage>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying buffers\n"); fflush(stderr);
    destroyAndClear(buffers, [](VkDevice d, uintptr_t h){ vkDestroyBuffer(d, reinterpret_cast<VkBuffer>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: freeing device memories\n"); fflush(stderr);
    destroyAndClear(deviceMemories, [](VkDevice d, uintptr_t h){ vkFreeMemory(d, reinterpret_cast<VkDeviceMemory>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying descriptorPools\n"); fflush(stderr);
    destroyAndClear(descriptorPools, [](VkDevice d, uintptr_t h){ vkDestroyDescriptorPool(d, reinterpret_cast<VkDescriptorPool>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying descriptorSetLayouts\n"); fflush(stderr);
    destroyAndClear(descriptorSetLayouts, [](VkDevice d, uintptr_t h){ vkDestroyDescriptorSetLayout(d, reinterpret_cast<VkDescriptorSetLayout>(h), nullptr); });

    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying renderPasses\n"); fflush(stderr);
    destroyAndClear(renderPasses, [](VkDevice d, uintptr_t h){ vkDestroyRenderPass(d, reinterpret_cast<VkRenderPass>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying semaphores\n"); fflush(stderr);
    destroyAndClear(semaphores, [](VkDevice d, uintptr_t h){ vkDestroySemaphore(d, reinterpret_cast<VkSemaphore>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying fences\n"); fflush(stderr);
    destroyAndClear(fences, [](VkDevice d, uintptr_t h){ vkDestroyFence(d, reinterpret_cast<VkFence>(h), nullptr); });
    fprintf(stderr, "[VulkanResourceManager] cleanup: destroying commandPools\n"); fflush(stderr);
    destroyAndClear(commandPools, [](VkDevice d, uintptr_t h){ vkDestroyCommandPool(d, reinterpret_cast<VkCommandPool>(h), nullptr); });

    // Descriptor sets are implicitly freed with pools; just clear the map
    {
        std::lock_guard<std::mutex> lk(mtx);
        descriptorSets.clear();
    }
}
