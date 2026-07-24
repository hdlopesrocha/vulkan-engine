#pragma once

#include <imgui.h>
#include <vulkan/vulkan.h>
#include <backends/imgui_impl_vulkan.h>
#include <unordered_map>
#include <mutex>
#include <vector>

class VulkanApp;

// Central manager for ImGui texture descriptors (ImTextureID).
//
// Each entry maps a (VkImageView, VkSampler) pair to an ImTextureID
// (which is a VkDescriptorSet in the Vulkan backend) created via
// ImGui_ImplVulkan_AddTexture.  The manager is the single source of
// truth for descriptor lifetimes.
//
// Lifecycle:
//   - getOrCreate() lazily creates a descriptor if none exists for
//     the given (view, sampler) pair.
//   - release() explicitly removes a cached entry and frees its
//     descriptor set via ImGui_ImplVulkan_RemoveTexture.
//   - invalidateAll() clears the cache without freeing descriptors.
//     Call this before destroying the ImGui descriptor pool during
//     swapchain recreation – the pool destruction implicitly frees
//     all descriptor sets.
//   - shutdown() destroys every cached descriptor (via deferred
//     RemoveTexture) and clears the cache.  Call this during app
//     shutdown while the ImGui backend is still alive.
class ImTextureManager {
public:
    ImTextureManager() = default;
    ~ImTextureManager() = default;

    // Non-copyable, movable
    ImTextureManager(const ImTextureManager&) = delete;
    ImTextureManager& operator=(const ImTextureManager&) = delete;
    ImTextureManager(ImTextureManager&&) = delete;
    ImTextureManager& operator=(ImTextureManager&&) = delete;

    // Return an ImTextureID for the given (view, sampler) pair.
    // Creates a new descriptor via ImGui_ImplVulkan_AddTexture if
    // none is cached.  Thread-safe.
    ImTextureID getOrCreate(VkImageView view, VkSampler sampler);

    // Explicitly remove a cached entry and call
    // ImGui_ImplVulkan_RemoveTexture on it.  Use this for per-frame
    // transient descriptors whose image views change each frame.
    // Thread-safe.
    void release(VkImageView view, VkSampler sampler);

    // Clear the cache without freeing any descriptor sets.
    // Safe to call when the descriptor pool is about to be destroyed
    // (e.g. during swapchain recreation) since pool destruction
    // implicitly reclaims all descriptor sets.  Thread-safe.
    void invalidateAll();

    // Destroy all cached descriptors via deferred RemoveTexture and
    // clear the cache.  The app pointer is used for
    // deferDestroyUntilAllPending() to ensure GPU-completion safety.
    // Thread-safe.
    void shutdown(VulkanApp* app);

    // Return the number of cached entries (for debugging).
    size_t size() const;

private:
    struct Key {
        VkImageView view;
        VkSampler sampler;

        bool operator==(const Key& o) const {
            return view == o.view && sampler == o.sampler;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            return std::hash<VkImageView>{}(k.view) ^
                   (std::hash<VkSampler>{}(k.sampler) << 1);
        }
    };

    std::unordered_map<Key, ImTextureID, KeyHash> m_cache;
    mutable std::mutex m_mutex;
};
