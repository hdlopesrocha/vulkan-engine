#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include <optional>

class VulkanResourceManager {
public:
    VulkanResourceManager() {
        // some std::unordered_map implementations start with zero buckets
        // which can trigger modulo-by-zero or free-of-sentinel bugs when the
        // first insertion occurs.  Pre-reserve a small number of buckets to
        // avoid these pitfalls and ensure safe use of operator[]/reserve/rehash.
        deviceMemories.reserve(1);
        images.reserve(1);
        imageViews.reserve(1);
        samplers.reserve(1);
        framebuffers.reserve(1);
        buffers.reserve(1);
        pipelines.reserve(1);
        pipelineLayouts.reserve(1);
        shaderModules.reserve(1);
        descriptorPools.reserve(1);
        descriptorSets.reserve(1);
        descriptorSetLayouts.reserve(1);
        renderPasses.reserve(1);
        semaphores.reserve(1);
        fences.reserve(1);
        commandPools.reserve(1);
    }
    ~VulkanResourceManager() = default;

    // All add/remove methods accept an optional description string identifying where the object was created.
    void addDeviceMemory(VkDeviceMemory mem, const char* desc = nullptr);
    void addImage(VkImage img, const char* desc = nullptr);
    void addImageView(VkImageView iv, const char* desc = nullptr);
    void addSampler(VkSampler s, const char* desc = nullptr);
    void addFramebuffer(VkFramebuffer fb, const char* desc = nullptr);
    void addRenderPass(VkRenderPass rp, const char* desc = nullptr);
    void addSemaphore(VkSemaphore s, const char* desc = nullptr);
    void addFence(VkFence f, const char* desc = nullptr);
    void addCommandPool(VkCommandPool cp, const char* desc = nullptr);
    void addBuffer(VkBuffer b, const char* desc = nullptr);
    void addPipeline(VkPipeline p, const char* desc = nullptr);
    void addPipelineLayout(VkPipelineLayout pl, const char* desc = nullptr);
    void addShaderModule(VkShaderModule m, const char* desc = nullptr);
    void addDescriptorPool(VkDescriptorPool dp, const char* desc = nullptr);
    void addDescriptorSet(VkDescriptorSet ds, const char* desc = nullptr);
    void addDescriptorSetLayout(VkDescriptorSetLayout dsl, const char* desc = nullptr);

    // Query a stored resource by handle
    struct Entry { VkObjectType type; std::string desc; };
    std::optional<Entry> find(uintptr_t handle) const;

    // Per-type map accessors (templated alias: handle type + object type)
    template<typename HandleT>
    using ResourceMap = std::unordered_map<uintptr_t, std::pair<HandleT, std::string>>;

    const ResourceMap<VkDeviceMemory> &getDeviceMemoryMap() const;
    const ResourceMap<VkImage> &getImageMap() const;
    const ResourceMap<VkImageView> &getImageViewMap() const;
    const ResourceMap<VkSampler> &getSamplerMap() const;
    const ResourceMap<VkFramebuffer> &getFramebufferMap() const;
    const ResourceMap<VkBuffer> &getBufferMap() const;
    const ResourceMap<VkPipeline> &getPipelineMap() const;
    const ResourceMap<VkPipelineLayout> &getPipelineLayoutMap() const;
    const ResourceMap<VkShaderModule> &getShaderModuleMap() const;
    const ResourceMap<VkDescriptorPool> &getDescriptorPoolMap() const;
    const ResourceMap<VkDescriptorSet> &getDescriptorSetMap() const;
    const ResourceMap<VkDescriptorSetLayout> &getDescriptorSetLayoutMap() const;
    const ResourceMap<VkRenderPass> &getRenderPassMap() const;
    const ResourceMap<VkSemaphore> &getSemaphoreMap() const;
    const ResourceMap<VkFence> &getFenceMap() const;
    const ResourceMap<VkCommandPool> &getCommandPoolMap() const;

    // Cleanup all resources in a safe deterministic order. The device must be valid.
    void cleanup(VkDevice device);

    // Remove methods (called when an owner explicitly destroys a handle)
    void removeDeviceMemory(VkDeviceMemory mem);
    void removeImage(VkImage img);
    void removeImageView(VkImageView iv);
    void removeSampler(VkSampler s);
    void removeFramebuffer(VkFramebuffer fb);
    void removeBuffer(VkBuffer b);
    void removePipeline(VkPipeline p);
    void removePipelineLayout(VkPipelineLayout pl);
    void removeShaderModule(VkShaderModule m);
    void removeDescriptorPool(VkDescriptorPool dp);
    void removeDescriptorSet(VkDescriptorSet ds);
    void removeDescriptorSetLayout(VkDescriptorSetLayout dsl);
    void removeRenderPass(VkRenderPass rp);
    void removeSemaphore(VkSemaphore s);
    void removeFence(VkFence f);
    void removeCommandPool(VkCommandPool cp);

private:
    mutable std::mutex mtx;
    // Per-type maps: handle -> (VkObjectType, description)
    ResourceMap<VkDeviceMemory> deviceMemories;
    ResourceMap<VkImage> images;
    ResourceMap<VkImageView> imageViews;
    ResourceMap<VkSampler> samplers;
    ResourceMap<VkFramebuffer> framebuffers;
    ResourceMap<VkBuffer> buffers;
    ResourceMap<VkPipeline> pipelines;
    ResourceMap<VkPipelineLayout> pipelineLayouts;
    ResourceMap<VkShaderModule> shaderModules;
    ResourceMap<VkDescriptorPool> descriptorPools;
    ResourceMap<VkDescriptorSet> descriptorSets;
    ResourceMap<VkDescriptorSetLayout> descriptorSetLayouts;
    ResourceMap<VkRenderPass> renderPasses;
    ResourceMap<VkSemaphore> semaphores;
    ResourceMap<VkFence> fences;
    ResourceMap<VkCommandPool> commandPools;

public:
};
