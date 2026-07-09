#pragma once

#include "../VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <stdexcept>

// Utility that wraps the common VkDescriptorSetLayout /
// VkDescriptorPool / VkDescriptorSet creation cycle into a few
// lines.  Each renderer that needs its own descriptor resources
// creates a local DescriptorAllocator instead of repeating the
// ~40-line boilerplate.
//
// Usage pattern:
//
//   DescriptorAllocator descAlloc{device, app};
//
//   VkDescriptorSetLayout layout = descAlloc.createLayout(bindings, flags, name);
//
//   VkDescriptorPool pool = descAlloc.createPool(poolSizes, maxSets, flags, name);
//
//   VkDescriptorSet set;
//   descAlloc.allocateSets(pool, layout, 1, &set, name);
//
// All created objects are automatically registered with
// VulkanResourceManager for lifetime tracking.  On failure a
// std::runtime_error is thrown (or std::cerr + VK_NULL_HANDLE if
// a createLayout returns null-handle).

struct DescriptorAllocator {
    VkDevice      device;
    VulkanApp*    app;

    DescriptorAllocator(VkDevice device_, VulkanApp* app_)
        : device(device_), app(app_) {}

    // ── Layout ──────────────────────────────────────────────────────────
    VkDescriptorSetLayout createLayout(
        const VkDescriptorSetLayoutBinding* bindings,
        uint32_t bindingCount,
        VkDescriptorSetLayoutCreateFlags flags = 0,
        const VkDescriptorBindingFlags* bindingFlags = nullptr,
        const char* name = "")
    {
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.flags = flags;
        info.bindingCount = bindingCount;
        info.pBindings = bindings;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
        if (bindingFlags) {
            flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            flagsInfo.bindingCount = bindingCount;
            flagsInfo.pBindingFlags = bindingFlags;
            info.pNext = &flagsInfo;
        }

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error(std::string("DescriptorAllocator: createLayout failed - ") + (name ? name : "unnamed"));
        }
        if (name) app->resources.addDescriptorSetLayout(layout, name);
        return layout;
    }

    // ── Pool ────────────────────────────────────────────────────────────
    VkDescriptorPool createPool(
        const VkDescriptorPoolSize* poolSizes,
        uint32_t poolSizeCount,
        uint32_t maxSets,
        VkDescriptorPoolCreateFlags flags = 0,
        const char* name = nullptr)
    {
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.poolSizeCount = poolSizeCount;
        info.pPoolSizes = poolSizes;
        info.maxSets = maxSets;
        info.flags = flags;

        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &info, nullptr, &pool) != VK_SUCCESS) {
            throw std::runtime_error(std::string("DescriptorAllocator: createPool failed - ") + (name ? name : "unnamed"));
        }
        if (name) app->resources.addDescriptorPool(pool, name);
        return pool;
    }

    // ── Single-set allocation ───────────────────────────────────────────
    VkDescriptorSet allocateSet(
        VkDescriptorPool pool,
        VkDescriptorSetLayout layout,
        const char* name = nullptr)
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;

        VkResult res = app->allocateDescriptorSetsThreadSafe(&alloc, &set);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(std::string("DescriptorAllocator: allocateSet failed - ") + (name ? name : "unnamed"));
        }
        if (name) {
            app->resources.addDescriptorSet(set, name);
        }
        return set;
    }

    // ── Multi-set allocation (same layout for all sets) ─────────────────
    void allocateSets(
        VkDescriptorPool pool,
        VkDescriptorSetLayout layout,
        uint32_t count,
        VkDescriptorSet* outSets,
        const char* name = nullptr)
    {
        std::vector<VkDescriptorSetLayout> layouts(count, layout);
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = count;
        alloc.pSetLayouts = layouts.data();

        VkResult res = app->allocateDescriptorSetsThreadSafe(&alloc, outSets);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(std::string("DescriptorAllocator: allocateSets failed - ") + (name ? name : "unnamed"));
        }
        if (name) {
            for (uint32_t i = 0; i < count; ++i) {
                app->resources.addDescriptorSet(outSets[i], name);
            }
        }
    }

    // ── Multi-set allocation (array of differing layouts) ──────────────
    void allocateSets(
        VkDescriptorPool pool,
        uint32_t count,
        const VkDescriptorSetLayout* layouts,
        VkDescriptorSet* outSets,
        const char* name = nullptr)
    {
        VkDescriptorSetAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = count;
        alloc.pSetLayouts = layouts;

        VkResult res = app->allocateDescriptorSetsThreadSafe(&alloc, outSets);
        if (res != VK_SUCCESS) {
            throw std::runtime_error(std::string("DescriptorAllocator: allocateSets failed - ") + (name ? name : "unnamed"));
        }
        if (name) {
            for (uint32_t i = 0; i < count; ++i) {
                app->resources.addDescriptorSet(outSets[i], name);
            }
        }
    }
};