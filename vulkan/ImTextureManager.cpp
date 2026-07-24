#include "ImTextureManager.hpp"
#include "VulkanApp.hpp"

ImTextureID ImTextureManager::getOrCreate(VkImageView view, VkSampler sampler) {
    if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE)
        return 0;

    std::lock_guard<std::mutex> lock(m_mutex);
    Key key{view, sampler};
    auto it = m_cache.find(key);
    if (it != m_cache.end())
        return it->second;

    ImTextureID id = (ImTextureID)ImGui_ImplVulkan_AddTexture(
        sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_cache.emplace(key, id);
    return id;
}

void ImTextureManager::release(VkImageView view, VkSampler sampler) {
    if (view == VK_NULL_HANDLE)
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    Key key{view, sampler};
    auto it = m_cache.find(key);
    if (it == m_cache.end())
        return;

    VkDescriptorSet ds = (VkDescriptorSet)it->second;
    if (ds != VK_NULL_HANDLE)
        ImGui_ImplVulkan_RemoveTexture(ds);
    m_cache.erase(it);
}

void ImTextureManager::invalidateAll() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache.clear();
}

void ImTextureManager::shutdown(VulkanApp* app) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& [key, id] : m_cache) {
        VkDescriptorSet ds = (VkDescriptorSet)id;
        if (ds != VK_NULL_HANDLE && app) {
            // Defer removal so descriptors in flight finish before we free them
            app->deferDestroyUntilAllPending(
                [ds]() { ImGui_ImplVulkan_RemoveTexture(ds); });
        } else if (ds != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(ds);
        }
    }
    m_cache.clear();
}

size_t ImTextureManager::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cache.size();
}
