#include "Buffer.hpp"
#include "VulkanApp.hpp"
#include <cstring>
#include <thread>
#include <execinfo.h>
#include <fstream>

static const char* layoutName(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
        case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: return "COLOR_ATTACHMENT_OPTIMAL";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_READ_ONLY_OPTIMAL";
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC_OPTIMAL";
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "TRANSFER_DST_OPTIMAL";
        case VK_IMAGE_LAYOUT_PREINITIALIZED: return "PREINITIALIZED";
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "PRESENT_SRC_KHR";
        default: return "UNKNOWN";
    }
}

static std::string buildTimestamp;

static void loadBuildTimestamp() {
    std::ifstream file("build_timestamp.txt");
    if (file.is_open()) {
        std::getline(file, buildTimestamp);
    } else {
        buildTimestamp = "unknown";
    }
}

Buffer VulkanApp::createDeviceLocalBufferAsync(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, VkFence* outFence) {
    auto stagingAlloc = stagingRing.allocate(size);
    if (!stagingAlloc.mappedPtr) {
        // Ring buffer exhausted or fragmented — fall back to dedicated staging buffer
        Buffer stagingBuffer = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(stagingBuffer.mappedData, data, (size_t)size);

        Buffer gpuBuffer = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkFence fence = runSingleTimeCommandsAsyncOnTransfer([&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            vkCmdCopyBuffer(cmd, stagingBuffer.buffer, gpuBuffer.buffer, 1, &copyRegion);
        });

        deferDestroyUntilFence(fence, [dev = device, sb = stagingBuffer, this]() {
            if (sb.allocation && vma.allocator) {
                vmaDestroyBuffer(vma.allocator, sb.buffer, sb.allocation);
            } else {
                if (sb.buffer != VK_NULL_HANDLE) {
                    if (resources.removeBuffer(sb.buffer)) vkDestroyBuffer(dev, sb.buffer, nullptr);
                }
                if (sb.memory != VK_NULL_HANDLE) {
                    if (resources.removeDeviceMemory(sb.memory)) vkFreeMemory(dev, sb.memory, nullptr);
                }
            }
        });

        if (outFence) *outFence = fence;
        return gpuBuffer;
    }

    memcpy(stagingAlloc.mappedPtr, data, (size_t)size);

    Buffer gpuBuffer = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBuffer stagingBuf = stagingRing.buffer();
    VkFence fence = runSingleTimeCommandsAsyncOnTransfer([&](VkCommandBuffer cmd) {
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = stagingAlloc.offset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, stagingBuf, gpuBuffer.buffer, 1, &copyRegion);
    });

    // Schedule ring buffer region release via deferred destruction
    {
        VkDeviceSize off = stagingAlloc.offset;
        deferDestroyUntilFence(fence, [this, off, sz = size]() {
            stagingRing.release(off, sz);
        });
    }

    if (outFence) *outFence = fence;
    return gpuBuffer;
}
#include <cstdint>
#include <atomic>

#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <unordered_set>
#include <cstdio>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <mutex>
#include <functional>
#include <vector>
#include <limits>

// Async submission bookkeeping
std::mutex pendingCmdMutex;
std::vector<std::pair<VkCommandBuffer,VkFence>> pendingCommandBuffers;
// Global submit counter and mapping to correlate vkQueueSubmit failures with commandbuffers
std::atomic<uint64_t> g_submitCounter{1};
std::unordered_map<VkCommandBuffer, uint64_t> g_cmdSubmitMap;
// Map command buffers to the command pool they were allocated from.
// This allows allocating per-thread temporary pools for async recording
// and freeing/destroying the correct pool when work completes.
std::unordered_map<VkCommandBuffer, VkCommandPool> commandBufferPoolMap;
// Optional per-command-buffer allocation backtrace to help correlate
// failing vkQueueSubmit() calls with the code that recorded/allocated
// the command buffer.
std::unordered_map<VkCommandBuffer, std::string> g_cmdBacktraces;

std::mutex extraSemaphoreMutex;
// Extra semaphores signaled by async submissions paired with the pipeline
// stage mask the frame submit should wait on for that semaphore.
std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> extraWaitSemaphores;
// semaphores scheduled for destruction paired with the frame fence they were associated with
std::vector<std::pair<VkSemaphore,VkFence>> semaphoresPendingDestroy;

// Deferred destruction bookkeeping (resource destroy callbacks paired with a fence to wait on)
std::mutex deferredDestroyMutex;
std::vector<std::pair<VkFence, std::function<void()>>> deferredDestroys;

// Buffers registered for automatic cleanup at final app shutdown.
std::mutex buffersPendingMutex;
std::vector<std::pair<VkBuffer, VkDeviceMemory>> buffersPendingAutoDestroy;
#include "VulkanApp.hpp"
#include "TextureArrayManager.hpp"
#include "../utils/FileReader.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include <cmath>
#include "VulkanResourceManager.hpp"

// ImGui glue pointer (set during initImGui)
VulkanApp* g_imguiVulkanApp = nullptr;

void setImGuiVulkanApp(VulkanApp* app) { g_imguiVulkanApp = app; }
VulkanApp* getImGuiVulkanApp() { return g_imguiVulkanApp; }

extern "C" void* ImGui_GetApp_C() { return (void*)getImGuiVulkanApp(); }
extern "C" void ImGui_SubmitCommandBufferAndWait_C(void* appPtr, VkCommandBuffer cb) {
    if (appPtr) ((VulkanApp*)appPtr)->submitCommandBufferAndWait(cb);
}
extern "C" VkResult ImGui_QueueWaitIdle_C(void* appPtr) {
    if (!appPtr) return VK_SUCCESS;
    return ((VulkanApp*)appPtr)->queueWaitIdle();
}
extern "C" VkResult ImGui_DeviceWaitIdle_C(void* appPtr) {
    if (!appPtr) return VK_SUCCESS;
    return ((VulkanApp*)appPtr)->deviceWaitIdle();
}

// C-callable bridge so external integration (e.g. ImGui backend) can
// record layout transitions using the application's tracked helper.
extern "C" void ImGui_RecordTransitionImageLayoutLayer_C(void* appPtr,
                                                         VkCommandBuffer commandBuffer,
                                                         VkImage image,
                                                         VkFormat format,
                                                         VkImageLayout oldLayout,
                                                         VkImageLayout newLayout,
                                                         uint32_t mipLevels,
                                                         uint32_t baseArrayLayer,
                                                         uint32_t layerCount) {
    if (!appPtr) return;
    VulkanApp* app = (VulkanApp*)appPtr;
    app->recordTransitionImageLayoutLayer(commandBuffer, image, format, oldLayout, newLayout, mipLevels, baseArrayLayer, layerCount);
   
}

void VulkanApp::initVulkan() {
    loadBuildTimestamp();
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createPipelineCache();
    vma.init(instance, physicalDevice, device);
    vmaReady = true;
    resources.setAllocator(vma.allocator);
    createSwapchain();
    createImageViews();
    createDescriptorSetLayout();
    createCommandPool();
    createDepthResources();
    commandBuffers = createCommandBuffers();
    createSyncObjects();
    initImGui();
}


void VulkanApp::createPipelineCache() {
    // Try to load existing cache data from disk
    std::vector<char> cacheData;
    std::ifstream file("pipeline_cache.bin", std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size_t size = static_cast<size_t>(file.tellg());
        file.seekg(0);
        cacheData.resize(size);
        file.read(cacheData.data(), static_cast<std::streamsize>(size));
        file.close();
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (!cacheData.empty()) {
        ci.initialDataSize = cacheData.size();
        ci.pInitialData = cacheData.data();
    }

    if (vkCreatePipelineCache(device, &ci, nullptr, &pipelineCache) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Warning: Failed to create pipeline cache, proceeding without" << std::endl;
        pipelineCache = VK_NULL_HANDLE;
    } else {
        printf("[VulkanApp] Created pipeline cache (%zu bytes loaded from disk)\n", cacheData.size());
    }
}

void VulkanApp::savePipelineCache() {
    if (pipelineCache == VK_NULL_HANDLE) {
        printf("[VulkanApp] Pipeline cache not available, skipping save\n");
        return;
    }

    size_t dataSize = 0;
    VkResult res = vkGetPipelineCacheData(device, pipelineCache, &dataSize, nullptr);
    if (res != VK_SUCCESS || dataSize == 0) {
        std::cerr << "[VulkanApp] Warning: Failed to get pipeline cache size" << std::endl;
        return;
    }

    std::vector<char> cacheData(dataSize);
    res = vkGetPipelineCacheData(device, pipelineCache, &dataSize, cacheData.data());
    if (res != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Warning: Failed to read pipeline cache data" << std::endl;
        return;
    }

    std::ofstream file("pipeline_cache.bin", std::ios::binary);
    if (file.is_open()) {
        file.write(cacheData.data(), static_cast<std::streamsize>(cacheData.size()));
        file.close();
        printf("[VulkanApp] Saved pipeline cache (%zu bytes)\n", cacheData.size());
    } else {
        std::cerr << "[VulkanApp] Warning: Failed to save pipeline cache to disk" << std::endl;
    }
}

void VulkanApp::requestClose() {
    if (window) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

// Utility: Generate vegetation instances using a compute shader
// vertexBuffer: input triangle mesh (positions, 3 floats per vertex)
// vertexCount: number of vertices in the buffer
// instancesPerTriangle: how many instances to generate per triangle
// outputBuffer: will be filled with instance positions (vec3)
// Returns: number of generated instances
// Now supports indexed geometry: pass indexBuffer and indexCount
uint32_t VulkanApp::generateVegetationInstancesCompute(
    VkBuffer vertexBuffer, uint32_t vertexCount,
    VkBuffer indexBuffer, uint32_t indexCount,
    uint32_t instancesPerTriangle,
    VkBuffer outputBuffer, uint32_t outputBufferSize, uint32_t seed) {
    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || outputBuffer == VK_NULL_HANDLE) return 0;
    if (indexCount < 3 || instancesPerTriangle == 0) return 0;

    uint32_t triCount = indexCount / 3;
    uint32_t expectedInstances = triCount * instancesPerTriangle;
    uint32_t expectedBytes = expectedInstances * sizeof(float) * 4; // shader writes vec4
    if (outputBufferSize < expectedBytes) {
        std::cerr << "[VulkanApp] generateVegetationInstancesCompute: outputBufferSize too small (" << outputBufferSize << " < " << expectedBytes << ")" << std::endl;
        return 0;
    }

    VkDevice device = getDevice();

    // Descriptor set layout: three storage buffers (vertices, indices, output instances)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descLayout) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Failed to create vegetation compute descriptor set layout" << std::endl;
        return 0;
    }

    // Push constant range matches shader Push struct (6 uints)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 6;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        std::cerr << "[VulkanApp] Failed to create vegetation compute pipeline layout" << std::endl;
        return 0;
    }

    // Load compute shader
    auto compCode = FileReader::readFile("shaders/vegetation_instance_gen.comp.spv");
    if (compCode.empty()) {
        std::cerr << "[VulkanApp] vegetation_instance_gen.comp.spv not found or empty" << std::endl;
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        return 0;
    }
    VkShaderModule compModule = createShaderModule(compCode);

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = compModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Failed to create vegetation compute pipeline" << std::endl;
        resources.removeShaderModule(compModule);
        vkDestroyShaderModule(device, compModule, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        return 0;
    }

    // Descriptor pool and set
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    // Allow freeing individual descriptor sets from this pool
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Failed to create descriptor pool for vegetation compute" << std::endl;
        vkDestroyPipeline(device, pipeline, nullptr);
        resources.removeShaderModule(compModule);
        vkDestroyShaderModule(device, compModule, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        return 0;
    }

    VkDescriptorSetAllocateInfo ainfo{};
    ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ainfo.descriptorPool = descPool;
    ainfo.descriptorSetCount = 1;
    ainfo.pSetLayouts = &descLayout;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &ainfo, &descSet) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Failed to allocate descriptor set for vegetation compute" << std::endl;
        resources.removeDescriptorPool(descPool);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        resources.removeShaderModule(compModule);
        vkDestroyShaderModule(device, compModule, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
        return 0;
    }

    VkDescriptorBufferInfo vbInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ibInfo{ indexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo outInfo{ outputBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &vbInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &ibInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &outInfo;

    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // Record and submit a single-time command buffer for compute dispatch
    runSingleTimeCommands([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

        // Push constants: instancesPerTriangle, vertexCount, indexCount, seed, baseTri, billboardCount
        uint32_t push[6];
        push[0] = instancesPerTriangle;
        push[1] = vertexCount;
        push[2] = indexCount;
        push[3] = seed;
        push[4] = 0u; // will be used as base triangle offset for chunked dispatch
        push[5] = 3u;

        if (triCount > 0) {
            // Query device limits for max dispatchable groups along X
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physicalDevice, &props);
            uint32_t maxGroupsX = props.limits.maxComputeWorkGroupCount[0];

            uint32_t remaining = triCount;
            uint32_t baseTri = 0;
            while (remaining > 0) {
                uint32_t thisGroups = remaining > maxGroupsX ? maxGroupsX : remaining;
                push[4] = baseTri;
                vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
                vkCmdDispatch(cmd, thisGroups, 1, 1);
                baseTri += thisGroups;
                remaining -= thisGroups;
            }

            // Ensure shader writes are visible to subsequent vertex input after all chunks
            VkBufferMemoryBarrier2 bufBarrier{};
            bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
            bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.buffer = outputBuffer;
            bufBarrier.offset = 0;
            bufBarrier.size = VK_WHOLE_SIZE;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &bufBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    });

    // Cleanup temporary Vulkan objects
    vkFreeDescriptorSets(device, descPool, 1, &descSet);
    resources.removeDescriptorPool(descPool);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    resources.removeShaderModule(compModule);
    vkDestroyShaderModule(device, compModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);

    return expectedInstances;
}

uint32_t VulkanApp::generateVegetationInstancesComputeAsync(
    VkBuffer vertexBuffer, uint32_t vertexCount,
    VkBuffer indexBuffer, uint32_t indexCount,
    uint32_t instancesPerTriangle,
    VkBuffer outputBuffer, uint32_t outputBufferSize, VkFence* outFence,
    uint32_t seed, uint32_t billboardCount) {
    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || outputBuffer == VK_NULL_HANDLE) return 0;
    if (indexCount < 3 || instancesPerTriangle == 0) return 0;

    uint32_t triCount = indexCount / 3;
    uint32_t expectedInstances = triCount * instancesPerTriangle;
    uint32_t expectedBytes = expectedInstances * sizeof(float) * 4;
    if (outputBufferSize < expectedBytes) {
        std::cerr << "[VulkanApp] generateVegetationInstancesComputeAsync: outputBufferSize too small (" << outputBufferSize << " < " << expectedBytes << ")" << std::endl;
        return 0;
    }

    VkDevice device = getDevice();

    // Ensure cached compute pipeline is ready (lazy init, thread-safe)
    if (!ensureVegetationComputePipeline()) {
        std::cerr << "[VulkanApp] Failed to ensure cached vegetation compute pipeline" << std::endl;
        return 0;
    }

    // Allocate a per-chunk descriptor set from the shared cached descriptor pool.
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ainfo{};
    ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ainfo.descriptorPool = vegComputeDescPool;
    ainfo.descriptorSetCount = 1;
    ainfo.pSetLayouts = &vegComputeDescSetLayout;
    {
        std::lock_guard<std::mutex> lk(vegComputeMutex);
    if (vkAllocateDescriptorSets(device, &ainfo, &descSet) != VK_SUCCESS) {
        std::cerr << "[VulkanApp::generateVegetationInstancesCompute] Failed to allocate descriptor set!" << std::endl;
        return 0;
    }
    std::cerr << "[RAW ALLOC] generateVegetationInstancesCompute: descSet=" << (void*)descSet << " pool=" << (void*)ainfo.descriptorPool << std::endl;
    }

    VkDescriptorBufferInfo vbInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ibInfo{ indexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo outInfo{ outputBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &vbInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &ibInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1;
    writes[2].pBufferInfo = &outInfo;

    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // Allocate a command buffer (temporary pool) for async dispatch
    VkCommandBuffer cmd = allocatePrimaryCommandBuffer();

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegComputePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vegComputePipelineLayout, 0, 1, &descSet, 0, nullptr);

    // Acquire vertex/index buffers for compute access. These were uploaded
    // via createDeviceLocalBuffer in a SEPARATE command buffer. On RADV,
    // inter-CB visibility requires an explicit acquire barrier even though
    // the transfer CB completed synchronously (fence wait).
    {
        VkBufferMemoryBarrier2 acquireBarriers[2]{};
        acquireBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        acquireBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        acquireBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        acquireBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acquireBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        acquireBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acquireBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        acquireBarriers[0].buffer = vertexBuffer;
        acquireBarriers[0].offset = 0;
        acquireBarriers[0].size = VK_WHOLE_SIZE;

        acquireBarriers[1] = acquireBarriers[0];
        acquireBarriers[1].buffer = indexBuffer;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
        depInfo.pBufferMemoryBarriers = acquireBarriers;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    uint32_t push[6];
    push[0] = instancesPerTriangle;
    push[1] = vertexCount;
    push[2] = indexCount;
    push[3] = seed;
    push[4] = 0u;
    push[5] = (billboardCount > 0) ? billboardCount : 1u;

    if (triCount > 0) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        uint32_t maxGroupsX = props.limits.maxComputeWorkGroupCount[0];

        uint32_t remaining = triCount;
        uint32_t baseTri = 0;
        while (remaining > 0) {
            uint32_t thisGroups = remaining > maxGroupsX ? maxGroupsX : remaining;
            push[4] = baseTri;
            vkCmdPushConstants(cmd, vegComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            vkCmdDispatch(cmd, thisGroups, 1, 1);
            baseTri += thisGroups;
            remaining -= thisGroups;
        }

        VkBufferMemoryBarrier2 bufBarrier{};
        bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bufBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        bufBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.buffer = outputBuffer;
        bufBarrier.offset = 0;
        bufBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &bufBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // End and submit to the vegetation queue. Signal a semaphore as well as a
    // fence: the fence gates CPU-side publication/destruction, while the
    // semaphore creates the required device-side dependency before graphics
    // reads the generated instance buffer.
    VkSemaphore completionSemaphore = VK_NULL_HANDLE;
    VkFence f = submitCommandBufferAsyncToQueue(cmd, vegetationQueue, &completionSemaphore);

    // Defer freeing only the per-chunk descriptor set; cached pipeline/pool persist.
    deferDestroyUntilFence(f, [device, descSet, this]() {
        if (descSet != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> lk(vegComputeMutex);
            std::cerr << "[DEFER FREE] Freeing vegCompute descSet=" << (void*)descSet << std::endl;
            vkFreeDescriptorSets(device, vegComputeDescPool, 1, &descSet);
        }
    });

    if (outFence) *outFence = f;
    return expectedInstances;
}

bool VulkanApp::ensureVegetationComputePipeline() {
    // Fast path: already initialized
    {
        std::lock_guard<std::mutex> lk(vegComputeMutex);
        if (vegComputePipeline != VK_NULL_HANDLE) return true;
    }

    VkDevice dev = getDevice();

    // Descriptor set layout: three storage buffers
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &descLayout) != VK_SUCCESS) {
        std::cerr << "[VulkanApp] Failed to create cached vegetation compute DS layout" << std::endl;
        return false;
    }

    // Push constant range: 6 uints
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 6;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &descLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(dev, &plInfo, nullptr, &pipeLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);
        std::cerr << "[VulkanApp] Failed to create cached vegetation compute pipeline layout" << std::endl;
        return false;
    }

    // Load compute shader
    auto compCode = FileReader::readFile("shaders/vegetation_instance_gen.comp.spv");
    if (compCode.empty()) {
        vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);
        std::cerr << "[VulkanApp] vegetation_instance_gen.comp.spv not found" << std::endl;
        return false;
    }
    VkShaderModule compModule = createShaderModule(compCode);

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = compModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage = stageInfo;
    cpInfo.layout = pipeLayout;

    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, pipelineCache, 1, &cpInfo, nullptr, &pipe) != VK_SUCCESS) {
        vkDestroyShaderModule(dev, compModule, nullptr);
        vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);
        std::cerr << "[VulkanApp] Failed to create cached vegetation compute pipeline" << std::endl;
        return false;
    }

    // Shared descriptor pool with enough capacity for concurrent async dispatches.
    // Use FREE_DESCRIPTOR_SET_BIT so individual sets can be freed without pool reset.
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3 * 64; // enough for 64 concurrent dispatches

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 64;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(dev, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, compModule, nullptr);
        vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
        vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);
        std::cerr << "[VulkanApp] Failed to create cached vegetation compute descriptor pool" << std::endl;
        return false;
    }

    // Commit cached state atomically
    {
        std::lock_guard<std::mutex> lk(vegComputeMutex);
        if (vegComputePipeline == VK_NULL_HANDLE) {
            vegComputePipeline       = pipe;
            vegComputePipelineLayout = pipeLayout;
            vegComputeDescSetLayout  = descLayout;
            vegComputeDescPool       = pool;
            vegComputeShaderModule   = compModule;

            resources.addPipeline(pipe, "VulkanApp: cachedVegetation ComputePipeline");
            resources.addPipelineLayout(pipeLayout, "VulkanApp: cachedVegetation ComputePipelineLayout");
            resources.addDescriptorSetLayout(descLayout, "VulkanApp: cachedVegetation ComputeDescSetLayout");
            resources.addDescriptorPool(pool, "VulkanApp: cachedVegetation ComputeDescPool");
            resources.addShaderModule(compModule, "VulkanApp: cachedVegetation ComputeShaderModule");
            std::cerr << "[VulkanApp] Cached vegetation compute pipeline ready: pipe=" << (void*)pipe << "\n";
        } else {
            // Another thread already initialized; clean up duplicates
            vkDestroyDescriptorPool(dev, pool, nullptr);
            vkDestroyPipeline(dev, pipe, nullptr);
            vkDestroyShaderModule(dev, compModule, nullptr);
            vkDestroyPipelineLayout(dev, pipeLayout, nullptr);
            vkDestroyDescriptorSetLayout(dev, descLayout, nullptr);
        }
    }
    return true;
}

void VulkanApp::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    // std::cerr << "[DEBUG] createImageViews: swapchainImageViews.size()=" << swapchainImageViews.size() << std::endl;
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapchainImageFormat;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
        resources.addImageView(swapchainImageViews[i], "VulkanApp: swapchainImageView");
    }
}

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    const char* sev = "UNKNOWN";
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) sev = "ERROR";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) sev = "WARNING";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) sev = "INFO";
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) sev = "VERBOSE";

    const char* tstr = "";
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) tstr = "GENERAL";
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) tstr = (strlen(tstr) ? ",VALIDATION" : "VALIDATION");
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) tstr = (strlen(tstr) ? ",PERFORMANCE" : "PERFORMANCE");

    // Suppress BestPractices messages — they come from third-party code
    // (ImGui) and SPIR-V tooling (WorkgroupSize deprecation) and cannot
    // be fixed without modifying external components.
    {
        const char* msg = (pCallbackData && pCallbackData->pMessage) ? pCallbackData->pMessage : "";
        if (strstr(msg, "BestPractices") != nullptr) return VK_FALSE;
        // Shadow passes use the same pipeline as color passes; the fragment
        // shader writes outColor but shadow rendering has no color attachment.
        // The write is correctly discarded — this is expected behavior.
        if (strstr(msg, "no VkRenderingInfo::pColorAttachments[0]") != nullptr) return VK_FALSE;
        // The 360° cubemap is rendered in an async command buffer on the same
        // queue and read by the main pass.  Barriers + semaphore ordering make
        // this safe, but the validation layer flags it as SYNC-HAZARD-READ-AFTER-WRITE
        // across command buffers (binding #11).  This is a known false positive.
        if (strstr(msg, "SYNC-HAZARD-READ-AFTER-WRITE") != nullptr && strstr(msg, "binding #11") != nullptr) return VK_FALSE;
        // Shader-OutputNotConsumed: vertex attribute declared in pipeline but
        // not read by the shader. Harmless — the GPU ignores unread inputs.
        if (strstr(msg, "Shader-OutputNotConsumed") != nullptr) return VK_FALSE;

    }

    // Only print WARNING and ERROR — suppress INFO/VERBOSE noise
    bool isWarningOrError = (messageSeverity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) != 0;
    if (isWarningOrError) {
        std::cerr << "validation(" << sev << ":" << tstr << ") [build: " << buildTimestamp << "] " << (pCallbackData && pCallbackData->pMessage ? pCallbackData->pMessage : "") << std::endl;
    }

    // Exit immediately on any real ERROR or WARNING so we can fix it.
    if (isWarningOrError) {
        _exit(1);
    }
    return VK_FALSE;
    return VK_FALSE;
}

// --- New helper methods for basic rendering ---
std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
    // VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME promoted to Vulkan 1.2 core — using vulkan12Features instead
    // VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME promoted to Vulkan 1.1 core
    // VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME promoted to Vulkan 1.2 core
};



void VulkanApp::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // explicit allow window resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Engine", nullptr, nullptr);
    // register resize callback so we can recreate the swapchain when user resizes window
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    // keyboard input is handled by the event system (KeyboardPublisher)
    // do not register a direct key callback here to avoid duplicate handling
}

void VulkanApp::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Intentionally left empty: keyboard input (including F11/ESC)
    // is handled by the event system (KeyboardPublisher -> EventManager).
    (void)window; (void)key; (void)scancode; (void)action; (void)mods;
}

void VulkanApp::toggleFullscreen() {
    if (!window) return;

    if (!isFullscreen) {
        // store windowed position and size
        glfwGetWindowPos(window, &windowedPosX, &windowedPosY);
        glfwGetWindowSize(window, &windowedWidth, &windowedHeight);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        if (!monitor) return;
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode) return;

        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        isFullscreen = true;
        return;
    }

    // restore windowed mode
    glfwSetWindowMonitor(window, nullptr, windowedPosX, windowedPosY, windowedWidth, windowedHeight, 0);
    isFullscreen = false;
}

void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
}

void VulkanApp::cleanup() {
    printf("[VulkanApp] cleanup start - device=%p\n", (void*)device);
    bool deviceLost = false;
    if (device != VK_NULL_HANDLE) {
        VkResult waitRes = deviceWaitIdle();
        if (waitRes == VK_ERROR_DEVICE_LOST) {
            deviceLost = true;
            throw std::runtime_error("[VulkanApp] cleanup: device lost detected, skipping explicit destroy callbacks\n");
        }
    }

    // Allow the derived app to release manager-owned Vulkan resources now while
    // the device, descriptor pools and command pool are still valid. This
    // ensures per-manager destructors can call Vulkan destroy functions (or
    // ImGui removal) before we tear down common objects below.
    clean();

    // Tear down the staging ring buffer while the device is still valid
    stagingRing.cleanup(device);

    // Destroy the upload timeline semaphore
    if (uploadTimeline != VK_NULL_HANDLE) {
        resources.removeSemaphore(uploadTimeline);
        vkDestroySemaphore(device, uploadTimeline, nullptr);
        uploadTimeline = VK_NULL_HANDLE;
    }
    // Process any deferred-destruction callbacks immediately so resources
    // scheduled with deferDestroyUntilAllPending() are released while the
    // device is still valid.
    processPendingCommandBuffers();

    // Do not destroy objects that are tracked by VulkanResourceManager here.
    // Let the centralized manager perform destruction while the device is still valid.
    // Process pending command buffers and run deferred destroys one more time,
    // then invoke the resource manager to cleanup tracked Vulkan objects.
    processPendingCommandBuffers();
    {
        std::lock_guard<std::mutex> dd(deferredDestroyMutex);
        for (auto &p : deferredDestroys) {
            if (!deviceLost) p.second();
        }
        deferredDestroys.clear();
    }
    // ImGui cleanup (must happen before destroying descriptor pools and device)
    printf("[VulkanApp] calling cleanupImGui() - imguiDescriptorPool=%p\n", (void*)imguiDescriptorPool);
    cleanupImGui();
    printf("[VulkanApp] cleanupImGui() returned\n");

    // Clear local semaphore handles now that ResourceManager destroyed tracked semaphores
    for (auto &s : imageAvailableSemaphores) s = VK_NULL_HANDLE;
    for (auto &s : renderFinishedSemaphores) s = VK_NULL_HANDLE;
    imageAvailableSemaphores.clear();
    renderFinishedSemaphores.clear();

    // Swapchain and surface teardown (not owned by resource manager)
    if (swapchain != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySwapchainKHR)vkGetInstanceProcAddr(instance, "vkDestroySwapchainKHR");
        if (fp) fp(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    if (surface != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(instance, "vkDestroySurfaceKHR");
        if (fp) fp(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    // Destroy window and terminate GLFW BEFORE destroying Vulkan instance
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();

    // Ensure any pending command buffers have been processed and deferred destruction callbacks
    // have been executed before destroying the device to avoid object-tracking warnings.
    // Wait for all pending command buffers (blocks until fences signal) so deferred destroys
    // that were enqueued with VK_NULL_HANDLE can run safely.
    waitForAllPendingCommandBuffers();
    // Process pending command buffers and deferred destroys once more to flush callbacks.
    processPendingCommandBuffers();
    // As a final safety measure, run any remaining deferred destroy callbacks now while
    // the device and related objects are still valid. This prevents vkDestroyDevice
    // object-tracking warnings for things that were missed.
    {
        std::lock_guard<std::mutex> dd(deferredDestroyMutex);
        for (auto &p : deferredDestroys) {
            if (!deviceLost) p.second();
        }
        deferredDestroys.clear();
    }
    // Final sweep of pending callbacks before destroying device
    processPendingCommandBuffers();
    {
        std::lock_guard<std::mutex> dd(deferredDestroyMutex);
        for (auto &p : deferredDestroys) {
            if (!deviceLost) p.second();
        }
        deferredDestroys.clear();
    }
    printf("[VulkanApp] about to vkDestroyDevice(device=%p)\n", (void*)device);
        resources.cleanup(device);
        vma.destroy();
        savePipelineCache();
        if (pipelineCache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device, pipelineCache, nullptr);
            pipelineCache = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE) {
        // Ensure the device is idle and no further commands are executing
        deviceWaitIdle();
        // Final sweep of pending callbacks
        processPendingCommandBuffers();
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    printf("[VulkanApp] vkDestroyDevice returned\n");

    if (debugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) func(instance, debugMessenger, nullptr);
        debugMessenger = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    printf("[VulkanApp] vkDestroyInstance returned\n");
}

void VulkanApp::initImGui() {
    // Create descriptor pool for ImGui
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * (uint32_t)std::size(pool_sizes);
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create ImGui descriptor pool!");
    }
    // Register ImGui descriptor pool
    resources.addDescriptorPool(imguiDescriptorPool, "VulkanApp: imguiDescriptorPool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    // Ensure a default font is present, then merge an icon font (FontAwesome) into it.
    // Place a TTF at fonts/fa-solid-900.ttf (free subset) if you want icons.
    {
        ImFontConfig defaultCfg;
        defaultCfg.SizePixels = 13.0f;
        io.Fonts->AddFontDefault(&defaultCfg);
    }
    {
        ImFontConfig fontCfg;
        fontCfg.MergeMode = true;
        fontCfg.PixelSnapH = true;
        static const ImWchar icons_ranges[] = { 0xF000, 0xF8FF, 0 };
        const char* iconFontPath = "fonts/fa-solid-900.ttf";
        FILE* f = fopen(iconFontPath, "rb");
        if (f) {
            fclose(f);
            io.Fonts->AddFontFromFileTTF(iconFontPath, 16.0f, &fontCfg, icons_ranges);
            printf("[ImGui] Merged icon font: %s\n", iconFontPath);
        } else {
            printf("[ImGui] Icon font not found at %s; using placeholder glyphs\n", iconFontPath);
        }
    }

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imguiDescriptorPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    init_info.Allocator = nullptr;
    init_info.MinAllocationSize = 1024 * 1024; // Pad to 1MB to suppress validation small-allocation warnings
    init_info.CheckVkResultFn = [](VkResult err) {
        if (err != VK_SUCCESS) {
            std::cerr << "[ImGui] Vulkan error: " << err << std::endl;
            abort();
        }
    };
    VkPipelineRenderingCreateInfo imguiPipelineRenderingInfo{};
    imguiPipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    imguiPipelineRenderingInfo.colorAttachmentCount = 1;
    imguiPipelineRenderingInfo.pColorAttachmentFormats = &swapchainImageFormat;
    imguiPipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    imguiPipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = imguiPipelineRenderingInfo;
    init_info.UseDynamicRendering = true;

    bool imguiInitOk = ImGui_ImplVulkan_Init(&init_info);
    // Expose this VulkanApp instance to the ImGui backend so it can route
    // synchronous submits/waits through our helpers to avoid threading races.
    setImGuiVulkanApp(this);
    printf("[ImGui] ImGui_ImplVulkan_Init returned %s\n", imguiInitOk ? "true" : "false");

    // Fonts are uploaded automatically by the backend on first NewFrame()
    if (!imguiInitOk) {
        printf("[ImGui] ERROR: ImGui_ImplVulkan_Init failed!\n");
    }
}

void VulkanApp::cleanupImGui() {
    if (imguiDescriptorPool != VK_NULL_HANDLE) {
        printf("[VulkanApp] cleanupImGui start - imguiDescriptorPool=%p device=%p\n", (void*)imguiDescriptorPool, (void*)device);
        deviceWaitIdle();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        // Unregister and destroy descriptor pool immediately to avoid leaks
        resources.removeDescriptorPool(imguiDescriptorPool);
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
        imguiDescriptorPool = VK_NULL_HANDLE;
        printf("[VulkanApp] cleanupImGui done\n");
    }
}

// Default empty implementations for overridable hooks declared in the header.
void VulkanApp::renderImGui() {
    // No-op default
}

void VulkanApp::postSubmit() {
    // No-op default
}




VkSurfaceFormatKHR VulkanApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    throw std::runtime_error("chooseSwapSurfaceFormat: required surface format VK_FORMAT_B8G8R8A8_SRGB/VK_COLOR_SPACE_SRGB_NONLINEAR_KHR not available");
}

VkPresentModeKHR VulkanApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    // If V-Sync is disabled, prefer IMMEDIATE mode for uncapped FPS (lowest latency, may tear)
    if (!vsyncEnabled) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                return availablePresentMode;
            }
        }
    }
    // With V-Sync enabled, prefer MAILBOX (triple-buffering, low latency, no tearing)
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) return availablePresentMode;
    }
    throw std::runtime_error("chooseSwapPresentMode: preferred present mode not available (no fallback allowed)");
}

void VulkanApp::setVSyncEnabled(bool enabled) {
    if (vsyncEnabled != enabled) {
        vsyncEnabled = enabled;
        vsyncChanged = true; // will trigger swapchain recreation on next frame
    }
}

VkExtent2D VulkanApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {WIDTH, HEIGHT};
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

void VulkanApp::createSwapchain() {
    // query support
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    if (formatCount != 0) vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (presentModeCount != 0) vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(presentModes);
    VkExtent2D extent = chooseSwapExtent(capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    auto fpCreateSwapchain = (PFN_vkCreateSwapchainKHR)vkGetInstanceProcAddr(instance, "vkCreateSwapchainKHR");
    if (fpCreateSwapchain == nullptr || fpCreateSwapchain(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        // try using device-level create
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
    }

    uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, nullptr);
    swapchainImages.resize(actualImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &actualImageCount, swapchainImages.data());
    // std::cerr << "[DEBUG] createSwapchain: swapchainImages.size()=" << swapchainImages.size() << std::endl;
    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;
}

void VulkanApp::createCommandPool() {
    // destroy existing pools (if any)
    if (commandPool != VK_NULL_HANDLE) {
        resources.removeCommandPool(commandPool);
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    // Destroy per-frame pools
    for (VkCommandPool& p : frameCommandPools) {
        if (p != VK_NULL_HANDLE) {
            resources.removeCommandPool(p);
            vkDestroyCommandPool(device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    }
    frameCommandPools.clear();
    if (transientCommandPool != VK_NULL_HANDLE) {
        resources.removeCommandPool(transientCommandPool);
        vkDestroyCommandPool(device, transientCommandPool, nullptr);
        transientCommandPool = VK_NULL_HANDLE;
    }

    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    // No RESET_COMMAND_BUFFER_BIT: we reset whole pools via vkResetCommandPool.
    poolInfo.flags = 0;

    // Keep commandPool for legacy allocation (may hold other transient buffers).
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
    resources.addCommandPool(commandPool, "VulkanApp: commandPool");

    // Create one per-frame pool so that vkResetCommandPool resets only the
    // current frame's command buffer without touching in-flight frames.
    const uint32_t numImages = static_cast<uint32_t>(swapchainImages.size());
    frameCommandPools.resize(numImages, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < numImages; ++i) {
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &frameCommandPools[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create per-frame command pool!");
        }
        resources.addCommandPool(frameCommandPools[i], "VulkanApp: frameCommandPool");
    }

    // secondary/transient pool for short-lived or async work
    VkCommandPoolCreateInfo transInfo{};
    transInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    transInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(device, &transInfo, nullptr, &transientCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create transient command pool!");
    }
    resources.addCommandPool(transientCommandPool, "VulkanApp: transientCommandPool");

    // Create dedicated command pools for vegetation and geometry (transient, no individual reset)
    VkCommandPoolCreateInfo vegPoolInfo{};
    vegPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    vegPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    vegPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(device, &vegPoolInfo, nullptr, &vegetationCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vegetation command pool!");
    }
    resources.addCommandPool(vegetationCommandPool, "VulkanApp: vegetationCommandPool");

    VkCommandPoolCreateInfo geoPoolInfo{};
    geoPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    geoPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    geoPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(device, &geoPoolInfo, nullptr, &geometryCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create geometry command pool!");
    }
    resources.addCommandPool(geometryCommandPool, "VulkanApp: geometryCommandPool");

    // Create a dedicated command pool for the transfer queue if available
    if (queueFamilyIndices.transferFamily.has_value()) {
        VkCommandPoolCreateInfo transferPoolInfo{};
        transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        transferPoolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();
        transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(device, &transferPoolInfo, nullptr, &transferCommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create transfer command pool!");
        }
        resources.addCommandPool(transferCommandPool, "VulkanApp: transferCommandPool");
    }

    // initialize async bookkeeping containers
    // pendingCommandBuffers and extraWaitSemaphores are guarded by mutexes
    pendingCommandBuffers.clear();
    extraWaitSemaphores.clear();
    semaphoresPendingDestroy.clear();
}

void VulkanApp::runSingleTimeCommands(const std::function<void(VkCommandBuffer)>& fn) {
    // Allocate from the shared transient command pool while holding
    // `transientPoolMutex` to avoid using the same pool concurrently from
    // multiple threads. Submissions and waits remain serialized by
    // `graphicsSubmitMutex` to prevent simultaneous use of the VkQueue.
    if (transientCommandPool == VK_NULL_HANDLE) {
        throw std::runtime_error("transientCommandPool not initialized in runSingleTimeCommands");
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = transientCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    // Serialize allocation, begin, recording and end under the same mutex so
    // the transient command pool is not used concurrently by multiple threads
    // while command buffers are being recorded. Submissions/waits remain
    // serialized separately by `graphicsSubmitMutex`.
    {
        std::lock_guard<std::mutex> poolLock(transientPoolMutex);
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffer in runSingleTimeCommands");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
            throw std::runtime_error("failed to begin command buffer in runSingleTimeCommands");
        }

        fn(cmd);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
            throw std::runtime_error("failed to end command buffer in runSingleTimeCommands");
        }
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        std::lock_guard<std::mutex> poolLock(transientPoolMutex);
        vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
        throw std::runtime_error("failed to create fence for single-time submit");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    {
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);
        // Assign a submit id for this submission for diagnostics
        uint64_t submitId = g_submitCounter.fetch_add(1);
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            g_cmdSubmitMap[cmd] = submitId;
        }

        // Promote pending layout updates before submit so validation sees
        // a populated authoritative layout for affected subresources.
        preApplyPendingLayoutsBeforeSubmit(cmd);

        VkResult submitRes = vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence);
        if (submitRes != VK_SUCCESS) {
            if (submitRes == VK_ERROR_DEVICE_LOST) {
                deviceLost.store(true);
                std::cerr << "[VulkanApp] runSingleTimeCommands: vkQueueSubmit returned VK_ERROR_DEVICE_LOST; registering fence and deferring command-buffer cleanup\n";
                resources.addFence(fence, "VulkanApp::runSingleTimeCommands: fence");
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    if (commandBufferPoolMap.find(cmd) == commandBufferPoolMap.end()) {
                        commandBufferPoolMap[cmd] = transientCommandPool;
                    }
                    pendingCommandBuffers.emplace_back(cmd, fence);
                }
                return;
            } else {
                vkDestroyFence(device, fence, nullptr);
                std::lock_guard<std::mutex> poolLock(transientPoolMutex);
                vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
                throw std::runtime_error("failed to submit command buffer in runSingleTimeCommands");
            }
        }
    }

    // Wait for completion of the submitted command buffer via the fence,
    // then apply pending layout updates and free the command buffer.
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Apply any pending layout updates recorded while recording this
    // single-time command buffer before freeing it (the GPU work is
    // complete because we waited on the fence).
    applyPendingLayoutUpdatesForCommandBuffer(cmd);

    {
        std::lock_guard<std::mutex> poolLock(transientPoolMutex);
        vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
    }
    vkDestroyFence(device, fence, nullptr);
}

VkFence VulkanApp::runSingleTimeCommandsAsync(const std::function<void(VkCommandBuffer)>& fn, VkSemaphore* outSemaphore) {
    if (transientCommandPool == VK_NULL_HANDLE) {
        throw std::runtime_error("transientCommandPool not initialized in runSingleTimeCommandsAsync");
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = transientCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> poolLock(transientPoolMutex);
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffer in runSingleTimeCommandsAsync");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, transientCommandPool, 1, &cmd);
            throw std::runtime_error("failed to begin command buffer in runSingleTimeCommandsAsync");
        }    
        fn(cmd);
        // Do NOT call vkEndCommandBuffer here; submitCommandBufferAsync will end and submit it.
    }

    // Record mapping so pending processing knows which pool to free from
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        commandBufferPoolMap[cmd] = transientCommandPool;
    }

    // Submit asynchronously and return the fence so the caller may track completion.
    VkFence fence = submitCommandBufferAsync(cmd, outSemaphore);
    return fence;
}

VkFence VulkanApp::runSingleTimeCommandsAsyncOnTransfer(const std::function<void(VkCommandBuffer)>& fn, VkSemaphore* outSemaphore) {
    // All transfers use the graphics queue. The dedicated transfer queue on
    // RADV/RENOIR causes GPU instability with buffer copies later read by
    // rendering. Same-queue transfers with pipeline barriers are always safe.
    return runSingleTimeCommandsAsync(fn, outSemaphore);
}

void VulkanApp::runSingleTimeCommandsOnTransfer(const std::function<void(VkCommandBuffer)>& fn) {
    // All transfers (sync and async) use the graphics queue. The dedicated
    // transfer queue on RADV/RENOIR causes GPU instability with buffer copies
    // that are later read by rendering — even with correct synchronization.
    // Same-queue transfers with pipeline barriers are always safe.
    runSingleTimeCommands(fn);
}

void VulkanApp::submitAndWait(const VkSubmitInfo* submits, uint32_t submitCount, VkFence fence) {
    {
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);
        vkQueueSubmit(graphicsQueue, submitCount, submits, fence);
    }
    if (fence == VK_NULL_HANDLE) {
        vkQueueWaitIdle(graphicsQueue);
    } else {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    }
}

// Wait only for the graphics queue to become idle.  This mirrors
// `runSingleTimeCommands` behavior and is usually sufficient for
// synchronization, avoiding the cost of `vkDeviceWaitIdle`.
VkResult VulkanApp::queueWaitIdle() {
    return vkQueueWaitIdle(graphicsQueue);
}

VkResult VulkanApp::deviceWaitIdle() {
    return vkDeviceWaitIdle(device);
}

void VulkanApp::waitForFrameFences() {
    // Copy current in-flight fences under graphicsSubmitMutex then wait on them.
    std::vector<VkFence> fences;
    {
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);
        fences = inFlightFences;
    }
    for (VkFence f : fences) {
        if (f != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX);
        }
    }
}

void VulkanApp::createImageWithVma(const VkImageCreateInfo& imageInfo, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& allocation, VkDeviceMemory& imageMemory, const char* debugName) {
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VmaAllocationInfo allocInfo;
    if (vmaCreateImage(vma.allocator, &imageInfo, &allocCI, &image, &allocation, &allocInfo) != VK_SUCCESS)
        throw std::runtime_error(std::string("failed to create image with VMA: ") + (debugName ? debugName : "unnamed"));

    imageMemory = allocInfo.deviceMemory;

    printf("[VulkanApp::createImageWithVma] created image=%p usage=0x%08x size=%ux%ux%u format=%d mipLevels=%u arrayLayers=%u name=%s\n",
        (void*)image, (unsigned int)imageInfo.usage, imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth,
        (int)imageInfo.format, imageInfo.mipLevels, imageInfo.arrayLayers,
        debugName ? debugName : "unnamed");

    resources.addImageVma(image, allocation, debugName ? debugName : "VulkanApp: image");

    // Initialize per-layer layout tracking (default UNDEFINED)
    {
        std::lock_guard<std::mutex> lk(imageLayoutMutex);
        uint32_t layers = imageInfo.arrayLayers;
        for (uint32_t l = 0; l < layers; ++l) {
            uint64_t key = ( (uint64_t)(uintptr_t)image << 32 ) | (uint64_t)l;
            imageLayerLayouts[key] = VK_IMAGE_LAYOUT_UNDEFINED;
        }
    }
}

void VulkanApp::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, uint32_t mipLevelCount, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& allocation, VkDeviceMemory& imageMemory, const char* debugName) {
    VkImageCreateInfo imageInfo{};
    std::memset(&imageInfo, 0, sizeof(imageInfo));
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = nullptr;
    imageInfo.flags = 0;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevelCount;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.queueFamilyIndexCount = 0;
    imageInfo.pQueueFamilyIndices = nullptr;

    createImageWithVma(imageInfo, properties, image, allocation, imageMemory, debugName);
}

void VulkanApp::destroyImageWithVma(VkImage image, VmaAllocation allocation, VkDeviceMemory imageMemory) {
    if (image == VK_NULL_HANDLE) return;
    if (allocation && vma.allocator) {
        resources.removeImage(image);
        vmaDestroyImage(vma.allocator, image, allocation);
    } else {
        if (image != VK_NULL_HANDLE) {
            resources.removeImage(image);
            vkDestroyImage(device, image, nullptr);
        }
        if (imageMemory != VK_NULL_HANDLE) {
            resources.removeDeviceMemory(imageMemory);
            vkFreeMemory(device, imageMemory, nullptr);
        }
    }
}

void VulkanApp::generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount, uint32_t baseArrayLayer) {
    // Existing blocking helper kept for convenience
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    // Use runSingleTimeCommands to serialize allocation/submit/free and avoid
    // concurrent command-pool/queue usage from multiple threads.
    runSingleTimeCommands([&](VkCommandBuffer cmd) {
        recordGenerateMipmaps(cmd, image, imageFormat, texWidth, texHeight, mipLevels, layerCount, baseArrayLayer);
    });

    // Update authoritative tracked layout for the affected layers after
    // performing the synchronous transition so future record-time callers
    // observe the correct oldLayout.
    {
        std::lock_guard<std::mutex> lk(imageLayoutMutex);
        for (uint32_t l = 0; l < layerCount; ++l) {
            uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)(baseArrayLayer + l);
            imageLayerLayouts[key] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
}

// Record mipmap generation commands into an existing command buffer (no begin/end or wait)
void VulkanApp::recordGenerateMipmaps(VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, uint32_t layerCount, uint32_t baseArrayLayer) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;
        uint32_t targetLayer = baseArrayLayer + layer;

        // ensure base level is in TRANSFER_DST_OPTIMAL before generating mips
        // Use recordTransitionImageLayoutLayer so the barrier is skipped if
        // the subresource is already in TRANSFER_DST_OPTIMAL (e.g. after a
        // compute-shader mip-prep barrier from TextureMixer). When the image
        // needs a real transition, the helper picks the correct access masks
        // from the effective old layout.
        recordTransitionImageLayoutLayer(commandBuffer, image, imageFormat,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         1, targetLayer, 1);

        for (uint32_t i = 1; i < mipLevels; i++) {
            // transition current mip level i to TRANSFER_DST_OPTIMAL from UNDEFINED
            barrier.subresourceRange.baseMipLevel = i;
            barrier.subresourceRange.baseArrayLayer = targetLayer;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

            {
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
            }

            // now transition previous level (i-1) to TRANSFER_SRC_OPTIMAL
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.subresourceRange.baseArrayLayer = targetLayer;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

            {
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
            }

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = targetLayer;
            blit.srcSubresource.layerCount = 1;

            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = { std::max(1, mipWidth / 2), std::max(1, mipHeight / 2), 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = targetLayer;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(commandBuffer,
                        image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit,
                        VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.baseArrayLayer = targetLayer;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

            {
            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(commandBuffer, &depInfo);
            }

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        // Transition last mip level for this layer to SHADER_READ_ONLY_OPTIMAL
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.subresourceRange.baseArrayLayer = targetLayer;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }
}
// Process any pending command buffers (free command buffers and fences when fence signaled)
void VulkanApp::processPendingCommandBuffers() {
    // Process deferred-destruction callbacks first to avoid inspecting fences that
    // we might destroy below. For callbacks registered with a specific fence,
    // execute them when that fence signals. For callbacks registered with
    // VK_NULL_HANDLE (wait-for-all), execute them only when there are no pending
    // command buffers remaining.
    {
        std::lock_guard<std::mutex> dd(deferredDestroyMutex);
        for (auto it = deferredDestroys.begin(); it != deferredDestroys.end(); ) {
            VkFence f = it->first;
            auto fn = it->second; // copy the function so we can call it safely
            bool canRun = false;
            if (f == VK_NULL_HANDLE) {
                // run when no pending async command buffers are outstanding
                // AND when all per-frame inFlight fences are signaled. This
                // prevents destroying resources that may still be referenced
                // by submitted render command buffers (which use inFlightFences).
                bool pendingEmpty = false;
                {
                    std::lock_guard<std::mutex> lk(pendingCmdMutex);
                    pendingEmpty = pendingCommandBuffers.empty();
                }
                if (pendingEmpty) {
                    bool allFramesIdle = true;
                    for (const VkFence &frameFence : inFlightFences) {
                        if (frameFence == VK_NULL_HANDLE) continue;
                        // If resource manager no longer tracks this fence the
                        // fence was destroyed elsewhere; treat it as signaled.
                        if (!resources.find((uintptr_t)frameFence).has_value()) continue;
                        VkResult st = vkGetFenceStatus(device, frameFence);
                        if (st != VK_SUCCESS) { allFramesIdle = false; break; }
                    }
                    if (allFramesIdle) canRun = true;
                }
            } else {
                // If the resource manager no longer tracks this fence the
                // fence has been destroyed out-of-band; treat the callback as
                // eligible to run to avoid calling into an invalid handle.
                if (!resources.find((uintptr_t)f).has_value()) {
                    canRun = true;
                } else {
                    VkResult st = vkGetFenceStatus(device, f);
                    if (st == VK_SUCCESS) canRun = true;
                }
            }
            if (canRun) {
                if (f == VK_NULL_HANDLE) {
                    std::cerr << "[PROCESS PENDING] Running deferDestroyUntilAllPending callback at queue size=" << deferredDestroys.size() << std::endl;
                }
                fn();
                it = deferredDestroys.erase(it);
            } else ++it;
        }
    }

    // Now process pending command buffers and free their command buffers when fences signal.
    // We first collect signaled entries while holding the pendingCmdMutex, then
    // perform frees/destroys without holding that mutex to avoid lock reentrancy.
    // To avoid long CPU spikes while still preventing unbounded resource buildup,
    // process up to `MAX_PENDING_FREE_PER_FRAME` signaled entries per-frame. If a
    // very large number of signaled entries appears at once, fall back to processing
    // them all to avoid allowing growth to spiral out of control.
    std::vector<std::pair<VkCommandBuffer, VkFence>> toFree;
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        for (auto it = pendingCommandBuffers.begin(); it != pendingCommandBuffers.end(); ) {
            VkCommandBuffer cmd = it->first;
            VkFence fence = it->second;
            // Guard vkGetFenceStatus by ensuring the fence is still tracked
            // by the resource manager. If not tracked, assume it was
            // destroyed and treat it as signaled so we can clean up.
            bool signaledOrGone = false;
            if (!resources.find((uintptr_t)fence).has_value()) {
                signaledOrGone = true;
                } else {
                VkResult st = vkGetFenceStatus(device, fence);
                if (st == VK_SUCCESS) signaledOrGone = true;
            }
            if (signaledOrGone) {
                toFree.emplace_back(cmd, fence);
                it = pendingCommandBuffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    const size_t MAX_PENDING_FREE_PER_FRAME = 4; // user-requested cap
    const size_t FORCE_PROCESS_ALL_THRESHOLD = 64; // if many signaled, clear backlog
    size_t processCount = toFree.size();
    if (processCount > MAX_PENDING_FREE_PER_FRAME && processCount <= FORCE_PROCESS_ALL_THRESHOLD) {
        processCount = MAX_PENDING_FREE_PER_FRAME;
    }

    // If we won't process all signaled entries this frame, re-insert the remainder
    // back into the pending list so they'll be handled in subsequent frames.
    if (processCount < toFree.size()) {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        for (size_t i = processCount; i < toFree.size(); ++i) {
            pendingCommandBuffers.emplace_back(toFree[i]);
        }
    }

    for (size_t i = 0; i < processCount; ++i) {
        VkCommandBuffer cmd = toFree[i].first;
        VkFence fence = toFree[i].second;
        // Apply any pending layout updates recorded for this command buffer
        // before freeing it so the authoritative tracked layout matches what
        // the GPU has actually executed.
        applyPendingLayoutUpdatesForCommandBuffer(cmd);

        // Free the command buffer using the correct originating pool (may destroy the pool)
        freeCommandBuffer(cmd);
        // Destroy and unregister the fence.  Hold deferredDestroyMutex to:
        //  1) run and remove any deferred callbacks registered for this fence,
        //     so that no future processPendingCommandBuffers call will call
        //     vkGetFenceStatus on a fence we are about to destroy, and
        //  2) make the removeFence+vkDestroyFence pair atomic with respect
        //     to the deferred-destroy loop which also calls vkGetFenceStatus.
        {
            std::lock_guard<std::mutex> dd(deferredDestroyMutex);
            for (auto dit = deferredDestroys.begin(); dit != deferredDestroys.end(); ) {
                if (dit->first == fence) {
                    auto fn = dit->second;
                    fn();
                    dit = deferredDestroys.erase(dit);
                } else {
                    ++dit;
                }
            }
            resources.removeFence(fence);
            vkDestroyFence(device, fence, nullptr);
        }
    }
}

void VulkanApp::applyPendingLayoutUpdatesForCommandBuffer(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;
    // Move pending updates out under the pending lock
    std::vector<PendingLayoutUpdate> updates;
    {
        std::lock_guard<std::mutex> plk(pendingLayoutMutex);
        auto it = commandBufferPendingLayouts.find(cmd);
        if (it == commandBufferPendingLayouts.end()) return;
        updates = std::move(it->second);
        commandBufferPendingLayouts.erase(it);
    }

    if (updates.empty()) return;

    // Apply updates into the authoritative map
    {
        std::lock_guard<std::mutex> lk(imageLayoutMutex);
        for (const auto &u : updates) {
            for (uint32_t l = 0; l < u.layerCount; ++l) {
                uint64_t key = ((uint64_t)(uintptr_t)u.image << 32) | (uint64_t)(u.baseArrayLayer + l);
                imageLayerLayouts[key] = u.newLayout;
            }
        }
    }
    {
        uint64_t submitId = 0;
        {
            std::lock_guard<std::mutex> lk(pendingCmdMutex);
            auto it = g_cmdSubmitMap.find(cmd);
            if (it != g_cmdSubmitMap.end()) submitId = it->second;
        }
    }
}

void VulkanApp::preApplyPendingLayoutsBeforeSubmit(VkCommandBuffer commandBuffer) {
    // Obtain the submit id (if any) for diagnostics *before* taking
    // pendingLayoutMutex so we maintain a consistent lock ordering
    // with callers that hold graphicsSubmitMutex.
    uint64_t submitId = 0;
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        auto it = g_cmdSubmitMap.find(commandBuffer);
        if (it != g_cmdSubmitMap.end()) submitId = it->second;
    }

    // Build a map of latest-seen pending updates for affected subresources.
    // "Latest" means the last recorded update for each key, which reflects
    // the final layout the GPU image will be in after the command buffer
    // finishes executing. Only this command buffer's own pending entries
    // are considered (other command buffers' entries are only safe to apply
    // when those buffers complete, via processPendingCommandBuffers).
    std::unordered_map<uint64_t, VkImageLayout> latest;

    {
        std::lock_guard<std::mutex> plk(pendingLayoutMutex);
        auto pit = commandBufferPendingLayouts.find(commandBuffer);
        if (pit != commandBufferPendingLayouts.end()) {
            for (const auto &u : pit->second) {
                for (uint32_t l = 0; l < u.layerCount; ++l) {
                    uint64_t key = ((uint64_t)(uintptr_t)u.image << 32) | (uint64_t)(u.baseArrayLayer + l);
                    // Always overwrite — iterate in order so the last entry wins.
                    latest[key] = u.newLayout;
                }
            }
        }

        // NOTE: intentionally do NOT fill gaps from other command buffers.
        // Promoting pending updates recorded in other command buffers into
        // the authoritative map here is optimistic: those other command
        // buffers may not have executed before this submit, so applying
        // their updates can make the tracked state diverge from the GPU
        // state and lead to validation errors. Only apply this command
        // buffer's own pending updates; other pending updates will be
        // applied when their command buffers complete.
    }

    if (latest.empty()) return;


    // Apply the collected latest updates into the authoritative map,
    // but first log each image/layer with its previous tracked layout.
    std::lock_guard<std::mutex> lk(imageLayoutMutex);
    for (const auto &p : latest) {
        uint64_t key = p.first;
        VkImage image = (VkImage)(uintptr_t)(key >> 32);
        uint32_t layer = (uint32_t)(key & 0xffffffff);
        VkImageLayout prev = VK_IMAGE_LAYOUT_UNDEFINED;
        auto it = imageLayerLayouts.find(key);
        if (it != imageLayerLayouts.end()) prev = it->second;
        auto entry = resources.find((uintptr_t)image);
        std::string desc = entry ? entry->desc : std::string("(unknown)");
#if 0
        std::cerr << "[VulkanApp] preApply submitId=" << submitId << " image=" << (void*)image
                  << " desc=" << desc
                  << " layer=" << layer
                  << " prev=" << layoutName(prev)
                  << " new=" << layoutName(p.second) << std::endl;
#endif
        imageLayerLayouts[p.first] = p.second;
    }
}

void VulkanApp::waitForAllPendingCommandBuffers() {
    std::vector<VkFence> fences;
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        fences.reserve(pendingCommandBuffers.size());
        for (auto &p : pendingCommandBuffers) fences.push_back(p.second);
    }
    if (!fences.empty()) {
        // Wait indefinitely until all tracked fences signal
        vkWaitForFences(device, static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max());
        // Clear out any completed entries
        processPendingCommandBuffers();
    }
}

// Throttle helper: if too many pending command buffers are queued, wait
// for them to complete before allowing more submissions. This avoids
// unbounded growth of pending resources which can trigger driver GPU hangs
// on some implementations when memory/queue pressure is high.
void VulkanApp::throttleIfTooManyPending() {
    const size_t MAX_PENDING = 128;
    size_t pending = 0;
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        pending = pendingCommandBuffers.size();
    }
    if (pending <= MAX_PENDING) return;
    // Wait for all currently pending command buffers to complete.
    waitForAllPendingCommandBuffers();
}

// Check whether a fence is currently tracked as pending
bool VulkanApp::isFencePending(VkFence fence) {
    std::lock_guard<std::mutex> lk(pendingCmdMutex);
    for (auto &p : pendingCommandBuffers) {
        if (p.second == fence) return true;
    }
    return false;
}


// Submit a pre-recorded command buffer asynchronously and return a fence that will be signaled on completion.
VkFence VulkanApp::submitCommandBufferAsync(VkCommandBuffer commandBuffer, VkSemaphore* outSemaphore) {
    // If the app has too many pending async submissions, block briefly
    // until some complete to avoid overwhelming the driver.
    throttleIfTooManyPending();
    // End command buffer here (caller recorded commands assumed)
    vkEndCommandBuffer(commandBuffer);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence for async submit");
    }
    // Register fence for async submit
    resources.addFence(fence, "VulkanApp::submitCommandBufferAsync: fence");

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (outSemaphore) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semInfo, nullptr, &semaphore) != VK_SUCCESS) {
            resources.removeFence(fence);
            vkDestroyFence(device, fence, nullptr);
            throw std::runtime_error("failed to create semaphore for async submit");
        }
        // Register semaphore for async submit
        resources.addSemaphore(semaphore, "VulkanApp::submitCommandBufferAsync: semaphore");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = (semaphore != VK_NULL_HANDLE) ? 1u : 0u;
    submitInfo.pSignalSemaphores = (semaphore != VK_NULL_HANDLE) ? &semaphore : nullptr;

    {
        // Serialize pre-apply and the subsequent vkQueueSubmit so the
        // authoritative tracked layouts reflect submission order. Acquire
        // `graphicsSubmitMutex` first to maintain consistent lock ordering.
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);

        // Double-use validation: check if this command buffer is already pending
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            for (const auto& cbf : pendingCommandBuffers) {
                if (cbf.first == commandBuffer) {
                    std::cerr << "[VulkanApp][ERROR] Attempted to submit command buffer " << (void*)commandBuffer << " which is already pending! Aborting submission to prevent device loss." << std::endl;
                    // Clean up and abort
                    if (semaphore != VK_NULL_HANDLE) {
                        resources.removeSemaphore(semaphore);
                        vkDestroySemaphore(device, semaphore, nullptr);
                    }
                    resources.removeFence(fence);
                    vkDestroyFence(device, fence, nullptr);
                    throw std::runtime_error("Double-use of command buffer detected");
                }
            }
        }

        // Assign a submit id for this submission to aid post-mortem correlation
        uint64_t submitId = g_submitCounter.fetch_add(1);
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            g_cmdSubmitMap[commandBuffer] = submitId;
        }

        // Promote pending layout updates before submit so validation sees
        // a populated authoritative layout for affected subresources.
        preApplyPendingLayoutsBeforeSubmit(commandBuffer);

        VkResult submitRes = vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence);
        if (submitRes != VK_SUCCESS) {
            if (submitRes == VK_ERROR_DEVICE_LOST) {
                // Mark global device lost state so other subsystems can adapt.
                deviceLost.store(true);
                std::cerr << "[VulkanApp] submitCommandBufferAsync: vkQueueSubmit returned VK_ERROR_DEVICE_LOST\n";
                // Print allocation backtrace (if available) to aid root-cause analysis
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    auto it = g_cmdBacktraces.find(commandBuffer);
                    if (it != g_cmdBacktraces.end()) {
                        std::cerr << "[VulkanApp] submit id=" << submitId << " allocation backtrace:\n" << it->second;
                    }
                }
                // Ensure the fence is tracked (it was added earlier).  Also
                // record the command-buffer -> pool mapping if missing so the
                // centralized cleanup can free the command buffer later.
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    if (commandBufferPoolMap.find(commandBuffer) == commandBufferPoolMap.end()) {
                        commandBufferPoolMap[commandBuffer] = transientCommandPool;
                    }
                    pendingCommandBuffers.emplace_back(commandBuffer, fence);
                }
                // Keep semaphore and fence registered; centralized cleanup handles them.
                return fence;
            }
            if (semaphore != VK_NULL_HANDLE) {
                resources.removeSemaphore(semaphore);
                vkDestroySemaphore(device, semaphore, nullptr);
            }
            resources.removeFence(fence);
            vkDestroyFence(device, fence, nullptr);
            throw std::runtime_error("failed to submit async command buffer");
        }
    }

    // Track command buffer+fence ownership so we can free command buffers later when fence signals
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        // If there is no explicit mapping for which pool the command buffer
        // came from, assume it was allocated from the transient pool.
        if (commandBufferPoolMap.find(commandBuffer) == commandBufferPoolMap.end()) {
            commandBufferPoolMap[commandBuffer] = transientCommandPool;
        }
        pendingCommandBuffers.emplace_back(commandBuffer, fence);
    }

    if (semaphore != VK_NULL_HANDLE && outSemaphore) {
        *outSemaphore = semaphore;
        // register the semaphore so drawFrame will wait on it and later clean it up
        // Use a conservative union of likely consumer stages when the
        // producing queue is the graphics/compute family.
        std::lock_guard<std::mutex> lk(extraSemaphoreMutex);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        extraWaitSemaphores.emplace_back(semaphore, waitStage);
    }

    return fence;
}

// Submit a pre-recorded command buffer asynchronously to a specific queue and return a fence that will be signaled on completion.
VkFence VulkanApp::submitCommandBufferAsyncToQueue(VkCommandBuffer commandBuffer, VkQueue targetQueue, VkSemaphore* outSemaphore) {
    // Throttle excessive outstanding submissions which can cause driver hangs
    // on some implementations when resources are exhausted.
    throttleIfTooManyPending();
    vkEndCommandBuffer(commandBuffer);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
        throw std::runtime_error("failed to create fence for async submit");
    resources.addFence(fence, "VulkanApp::submitCommandBufferAsyncToQueue: fence");

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (outSemaphore) {
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semInfo, nullptr, &semaphore) != VK_SUCCESS) {
            resources.removeFence(fence);
            vkDestroyFence(device, fence, nullptr);
            throw std::runtime_error("failed to create semaphore");
        }
        resources.addSemaphore(semaphore, "VulkanApp::submitCommandBufferAsyncToQueue: semaphore");
    }

    // Signal only binary semaphore if requested. Same-queue transfers don't
    // need timeline — queue submission order guarantees transfer-before-render.
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = (semaphore != VK_NULL_HANDLE) ? 1u : 0u;
    submitInfo.pSignalSemaphores = (semaphore != VK_NULL_HANDLE) ? &semaphore : nullptr;

    {
        // Serialize pre-apply and submission to maintain consistent ordering
        // Select per-queue mutex to avoid serializing transfer/graphics/compute
        std::mutex& submitMtx = (targetQueue == transferQueue) ? transferSubmitMutex :
                                (targetQueue == vegetationQueue) ? vegetationSubmitMutex :
                                graphicsSubmitMutex;
        std::lock_guard<std::mutex> lock(submitMtx);

        // Double-use validation: check if this command buffer is already pending
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            for (const auto& cbf : pendingCommandBuffers) {
                if (cbf.first == commandBuffer) {
                    std::cerr << "[VulkanApp][ERROR] Attempted to submit command buffer " << (void*)commandBuffer << " which is already pending! Aborting submission to prevent device loss." << std::endl;
                    if (semaphore != VK_NULL_HANDLE) {
                        resources.removeSemaphore(semaphore);
                        vkDestroySemaphore(device, semaphore, nullptr);
                    }
                    resources.removeFence(fence);
                    vkDestroyFence(device, fence, nullptr);
                    throw std::runtime_error("Double-use of command buffer detected");
                }
            }
        }

        // Assign a submit id for this submission to aid post-mortem correlation
        uint64_t submitId = g_submitCounter.fetch_add(1);
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            g_cmdSubmitMap[commandBuffer] = submitId;
        }
        std::cerr << "[VulkanApp] submitToQueue id=" << submitId << " cmd=" << (void*)commandBuffer << " fence=" << (void*)fence << " sem=" << (void*)semaphore << " targetQueue=" << (void*)targetQueue << std::endl;

       
        // Promote pending layout updates for this submission
        preApplyPendingLayoutsBeforeSubmit(commandBuffer);

        VkResult submitRes = vkQueueSubmit(targetQueue, 1, &submitInfo, fence);
        if (submitRes != VK_SUCCESS) {
            if (semaphore != VK_NULL_HANDLE) {
                if (submitRes == VK_ERROR_DEVICE_LOST) {
                    std::cerr << "[VulkanApp] submitCommandBufferAsyncToQueue: vkQueueSubmit returned VK_ERROR_DEVICE_LOST\n";
                } else {
                    resources.removeSemaphore(semaphore);
                    vkDestroySemaphore(device, semaphore, nullptr);
                }
            }
            if (submitRes == VK_ERROR_DEVICE_LOST) {
                deviceLost.store(true);
                // Print allocation backtrace (if available) to aid debugging
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    auto it = g_cmdBacktraces.find(commandBuffer);
                    if (it != g_cmdBacktraces.end()) {
                        std::cerr << "[VulkanApp] submitToQueue id=" << submitId << " allocation backtrace:\n" << it->second;
                    }
                }
                // Ensure the command buffer is tracked and deferred for cleanup
                // so we don't attempt to free it while the driver still
                // considers it pending.
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    if (commandBufferPoolMap.find(commandBuffer) == commandBufferPoolMap.end()) {
                        commandBufferPoolMap[commandBuffer] = transientCommandPool;
                    }
                    pendingCommandBuffers.emplace_back(commandBuffer, fence);
                }
                // Leave fence registered for centralized cleanup and return the fence so caller can still defer destroys.
                return fence;
            } else {
                resources.removeFence(fence);
                vkDestroyFence(device, fence, nullptr);
                throw std::runtime_error("failed to submit async command buffer to target queue");
            }
        }
    }

    // Track command buffer+fence ownership so we can free command buffers later when fence signals
    {
        std::lock_guard<std::mutex> lk(pendingCmdMutex);
        pendingCommandBuffers.emplace_back(commandBuffer, fence);
    }

    if (semaphore != VK_NULL_HANDLE && outSemaphore) {
        *outSemaphore = semaphore;

        std::lock_guard<std::mutex> lk(extraSemaphoreMutex);
        VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
        extraWaitSemaphores.emplace_back(semaphore, waitStage);
    }

    return fence;
}

void VulkanApp::submitCommandBufferAndWait(VkCommandBuffer commandBuffer) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fence for submitCommandBufferAndWait");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    {
        // Serialize pre-apply and submission to maintain consistent ordering
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);

        // Promote pending layout updates for this submission so validation
        // sees populated layouts for affected subresources.
        preApplyPendingLayoutsBeforeSubmit(commandBuffer);

        VkResult submitRes = vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence);
        if (submitRes != VK_SUCCESS) {
            if (submitRes == VK_ERROR_DEVICE_LOST) {
                deviceLost.store(true);
                std::cerr << "[VulkanApp] submitCommandBufferAndWait: vkQueueSubmit returned VK_ERROR_DEVICE_LOST\n";
                // Print allocation backtrace (if available) to help correlate
                // the failing submit with the recording site.
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    auto it = g_cmdBacktraces.find(commandBuffer);
                    if (it != g_cmdBacktraces.end()) {
                        std::cerr << "[VulkanApp] submitCommandBufferAndWait allocation backtrace:\n" << it->second;
                    }
                }
                // Register fence with resource tracker and defer freeing the
                // command buffer so we don't call vkFreeCommandBuffers while
                // the driver still considers it in use.
                resources.addFence(fence, "VulkanApp::submitCommandBufferAndWait: fence");
                {
                    std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
                    if (commandBufferPoolMap.find(commandBuffer) == commandBufferPoolMap.end()) {
                        commandBufferPoolMap[commandBuffer] = commandPool;
                    }
                    pendingCommandBuffers.emplace_back(commandBuffer, fence);
                }
                return;
            }
            vkDestroyFence(device, fence, nullptr);
            throw std::runtime_error("failed to submit command buffer in submitCommandBufferAndWait");
        }
    }

    // Wait for the fence to signal completion (avoids calling vkQueueWaitIdle)
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    // The command buffer has completed execution; apply any pending layout
    // updates that were recorded into it.
    applyPendingLayoutUpdatesForCommandBuffer(commandBuffer);
    vkDestroyFence(device, fence, nullptr);
}

void VulkanApp::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t arrayLayers) {
    if (image == VK_NULL_HANDLE) {
        throw std::runtime_error("transitionImageLayout called with VK_NULL_HANDLE image!");
    }
    auto aspectFromFormat = [](VkFormat fmt) -> VkImageAspectFlags {
        switch (fmt) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_X8_D24_UNORM_PACK32:
            case VK_FORMAT_D32_SFLOAT:
                return VK_IMAGE_ASPECT_DEPTH_BIT;
            case VK_FORMAT_D16_UNORM_S8_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                return VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
            default:
                return VK_IMAGE_ASPECT_COLOR_BIT;
        }
    };

    runSingleTimeCommands([&](VkCommandBuffer commandBuffer){
        // For whole-image synchronous transitions use baseArrayLayer=0 and
        // cover all array layers. Prefer the app-tracked layout if present.
        VkImageLayout effectiveOld = oldLayout;
        {
            std::lock_guard<std::mutex> lk(imageLayoutMutex);
            uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)0;
            auto it = imageLayerLayouts.find(key);
            if (it != imageLayerLayouts.end()) {
                VkImageLayout tracked = it->second;
                if (tracked != oldLayout) {
                    // std::cerr << "[VulkanApp] transitionImageLayoutLayer: image=" << (void*)image << " callerOld=" << (int)oldLayout << " trackedOld=" << (int)tracked << " -> using tracked" << std::endl;
                    effectiveOld = tracked;
                }
            } else {
                imageLayerLayouts[key] = oldLayout;
            }
        }

        if (effectiveOld > 7 && effectiveOld < 1000000000u) {
            std::cerr << "[VulkanApp] WARNING: invalid effectiveOld " << effectiveOld
                      << " for image " << (void*)image << ", clamping to UNDEFINED" << std::endl;
            effectiveOld = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // If the tracked layout already matches the requested final layout,
        // no barrier is required.
        if (effectiveOld == newLayout) {
            return;
        }

        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = effectiveOld;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectFromFormat(format);
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = arrayLayers;

        VkPipelineStageFlags2 sourceStage;
        VkPipelineStageFlags2 destinationStage;

        // Use the authoritative old layout (effectiveOld) when selecting
        // access masks and pipeline stages so they match barrier.oldLayout.
        if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if ((effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // Transition depth attachment -> shader read (for sampling depth textures)
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        barrier.srcStageMask = sourceStage;
        barrier.dstStageMask = destinationStage;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        // Debug: log the executed barrier and its selected masks/stages
        // std::cerr << "[VulkanApp] recordTransitionImageLayoutLayer: cmd=" << (void*)commandBuffer << " image=" << (void*)image << " old=" << (int)barrier.oldLayout << " new=" << (int)newLayout << " srcAccess=0x" << std::hex << (unsigned)barrier.srcAccessMask << " dstAccess=0x" << (unsigned)barrier.dstAccessMask << " srcStage=0x" << (unsigned)sourceStage << " dstStage=0x" << (unsigned)destinationStage << " aspect=0x" << (unsigned)barrier.subresourceRange.aspectMask << " baseLayer=" << (unsigned)barrier.subresourceRange.baseArrayLayer << " layerCount=" << (unsigned)barrier.subresourceRange.layerCount << std::endl;
        // Update authoritative tracked layout for the affected layers
        {
            std::lock_guard<std::mutex> lk(imageLayoutMutex);
            for (uint32_t l = 0; l < arrayLayers; ++l) {
                uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)l;
                imageLayerLayouts[key] = newLayout;
            }
        }
    });
}

    VkCommandBuffer VulkanApp::allocatePrimaryCommandBuffer() {
        // Allocate a temporary per-call command pool and allocate a primary
        // command buffer from it. This lets threads record in parallel without
        // sharing a single command pool. We track the mapping from command
        // buffer -> pool so we can free/destroy the correct pool later.
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandPool tempPool = VK_NULL_HANDLE;

        QueueFamilyIndices qfi = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = qfi.graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &tempPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create temporary command pool for async submit");
        }
        resources.addCommandPool(tempPool, "VulkanApp: tempAsyncCommandPool");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = tempPool;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
            resources.removeCommandPool(tempPool);
            vkDestroyCommandPool(device, tempPool, nullptr);
            throw std::runtime_error("failed to allocate command buffer for async submit");
        }

        // Track mapping so we can free using the correct pool later.
        {
            std::lock_guard<std::mutex> lk(pendingCmdMutex);
            commandBufferPoolMap[cmd] = tempPool;

            // Capture an allocation backtrace for this command buffer so
            // we can later correlate failing vkQueueSubmit() calls with
            // the code path that allocated/recorded the command buffer.
            void* bt_buf[32];
            int bt_n = backtrace(bt_buf, 32);
            char** bt_syms = backtrace_symbols(bt_buf, bt_n);
            std::string bt_str;
            for (int i = 1; i < bt_n; ++i) { // skip frame 0 (this function)
                if (bt_syms && bt_syms[i]) {
                    bt_str += bt_syms[i];
                    bt_str += "\n";
                }
            }
            if (bt_syms) free(bt_syms);
            g_cmdBacktraces[cmd] = bt_str;
        }
        return cmd;
    }

    void VulkanApp::freeCommandBuffer(VkCommandBuffer cmd) {
        if (cmd == VK_NULL_HANDLE) return;
        VkDevice dev = device;
        VkCommandPool pool = VK_NULL_HANDLE;
        // Extract mapping (if any)
        {
            std::lock_guard<std::mutex> lk(pendingCmdMutex);
            auto it = commandBufferPoolMap.find(cmd);
            if (it != commandBufferPoolMap.end()) {
                pool = it->second;
                commandBufferPoolMap.erase(it);
            }
        }
        if (pool == VK_NULL_HANDLE) pool = commandPool;

        // Free the command buffer from the appropriate pool.
        // The transientCommandPool is protected by transientPoolMutex
        // (used by runSingleTimeCommands / runSingleTimeCommandsAsync for
        // allocation). All other pools are protected by commandPoolMutex.
        if (pool == transientCommandPool) {
            std::lock_guard<std::mutex> lock(transientPoolMutex);
            vkFreeCommandBuffers(dev, pool, 1, &cmd);
        } else {
            std::lock_guard<std::mutex> lock(commandPoolMutex);
            vkFreeCommandBuffers(dev, pool, 1, &cmd);
        }

        // Remove any submit mapping for this command buffer (cleanup trace state)
        {
            std::lock_guard<std::mutex> lk(pendingCmdMutex);
            g_cmdSubmitMap.erase(cmd);
            g_cmdBacktraces.erase(cmd);
        }

        // If this was a temporary pool we created, destroy it now
        if (pool != VK_NULL_HANDLE && pool != commandPool && pool != transientCommandPool && pool != transferCommandPool) {
            resources.removeCommandPool(pool);
            vkDestroyCommandPool(dev, pool, nullptr);
        }
    }

    void VulkanApp::recordTransitionImageLayoutLayer(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount) {
        if (commandBuffer == VK_NULL_HANDLE) throw std::runtime_error("recordTransitionImageLayoutLayer called with VK_NULL_HANDLE commandBuffer");
        if (image == VK_NULL_HANDLE) throw std::runtime_error("recordTransitionImageLayoutLayer called with VK_NULL_HANDLE image");

            // If this image belongs to the swapchain, skip recording transitions
            // here — `drawFrame()` performs explicit swapchain image transitions
            // (UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR).
            // Recording additional transitions for swapchain images from other
            // modules can produce duplicate barriers and validation hazards.
            for (const auto &si : swapchainImages) {
                if (si == image) {
                    std::cerr << "[VulkanApp] recordTransitionImageLayoutLayer: skipping swapchain image=" << (void*)image << " (drawFrame handles swapchain transitions)" << std::endl;
                    return;
                }
            }

        // Determine authoritative oldLayout per image-layer. If the caller's
        // supplied oldLayout disagrees with the app-tracked layout, prefer the
        // app-tracked value to avoid validation-layer VUID-oldLayout-01197.
        VkImageLayout effectiveOld = oldLayout;
        VkImageLayout tracked = VK_IMAGE_LAYOUT_UNDEFINED;
        {
            std::lock_guard<std::mutex> lk(imageLayoutMutex);
            uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)baseArrayLayer;
            auto it = imageLayerLayouts.find(key);
            if (it != imageLayerLayouts.end()) {
                tracked = it->second;
                if (tracked != oldLayout) {
                    // Prefer the app-tracked layout when available. Recording
                    // a transition from the tracked layout reduces the chance
                    // of emitting barriers that claim an UNDEFINED oldLayout
                    // while the app (or previous submissions) know the image
                    // is already in a concrete layout. This helps avoid
                    // validation errors when callers pass VK_IMAGE_LAYOUT_UNDEFINED
                    // to indicate they don't know the current layout.
                    // KHR/EXT extension layouts have values in 1000000000+ range.
                    // Catch truly corrupted values (small positive numbers that aren't
                    // valid core layouts 0-5 or the KHR attachment layout 7).
                    effectiveOld = tracked;
                }
            } else {
                // Initialize tracking for this layer from caller's oldLayout.
                imageLayerLayouts[key] = oldLayout;
            }
        }

        // Guard against corrupted effectiveOld (from caller or tracked map)
        if (effectiveOld > 7 && effectiveOld < 1000000000u) {
            std::cerr << "[VulkanApp] WARNING: invalid effectiveOld " << effectiveOld
                      << " for image " << (void*)image << ", clamping to UNDEFINED" << std::endl;
            effectiveOld = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // If this command buffer has previously recorded layout updates for
        // the same image/layer, prefer the most recent pending value so
        // subsequent records within the same command buffer observe earlier
        // recorded operations. Use the most recent pending update (barrier or
        // tracked-only) because tracked entries can represent implicit
        // render-pass transitions that affect the effective layout.
        VkImageLayout pendingOld = VK_IMAGE_LAYOUT_UNDEFINED;
        {
            std::lock_guard<std::mutex> plk(pendingLayoutMutex);
            auto pit = commandBufferPendingLayouts.find(commandBuffer);
            if (pit != commandBufferPendingLayouts.end()) {
                auto &vec = pit->second;
                // Prefer the most recent pending update that was recorded as
                // an actual barrier. Tracked-only updates (isBarrier==false)
                // represent implicit render-pass finalLayouts and do not
                // correspond to an emitted VkImageMemoryBarrier; using them
                // as the effective old layout for a vkCmdPipelineBarrier can
                // lead to validation-layer mismatches. If no barrier-type
                // pending update exists, fall back to the most recent
                // pending update of any type.
                bool foundBarrier = false;
                // Prefer the most recent pending barrier recorded for this
                // command buffer. If an earlier vkCmdPipelineBarrier was
                // emitted in the same command buffer for the same image/layer,
                // the effective old layout for any subsequent barrier must
                // reflect that earlier pending barrier to avoid emitting a
                // second barrier whose oldLayout disagrees with the validation
                // layer's recorded state for this command buffer.
                for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
                    if (it->image == image && it->isBarrier) {
                        uint32_t pendingBase = it->baseArrayLayer;
                        uint32_t pendingCount = it->layerCount;
                        // Treat a pending update as applicable if it covers the
                        // requested base array layer (overlap). This handles
                        // cases where a previous barrier updated multiple
                        // layers but the current request targets a single
                        // layer inside that range.
                        if (baseArrayLayer >= pendingBase && baseArrayLayer < pendingBase + pendingCount) {
                            pendingOld = it->newLayout;
                            // A recorded barrier in the same command buffer is
                            // the authoritative subresource layout for any
                            // subsequent barrier in that command buffer.
                            // Always prefer it over global tracked state.
                            effectiveOld = it->newLayout;
                            foundBarrier = true;
                            break;
                        }
                    }
                }
                if (!foundBarrier) {
                    // If no barrier-type pending update exists, do NOT adopt
                    // tracked-only pending updates as the effective old layout.
                    // Tracked-only updates represent implicit render-pass
                    // finalLayouts and do not correspond to an emitted
                    // VkImageMemoryBarrier; using them here can cause
                    // validation-layer mismatches. Leave `effectiveOld`
                    // unchanged so we prefer the caller-supplied or
                    // authoritative tracked layout instead.
                    (void)pendingOld;
                }
            }
        }

        // If the authoritative (tracked) layout already equals the requested
        // final layout, there's nothing to emit.
        if (effectiveOld == newLayout) {
            // std::cerr << "[VulkanApp] recordTransitionImageLayoutLayer: cmd=" << (void*)commandBuffer << " image=" << (void*)image << " effectiveOld==new (" << (int)effectiveOld << "), skipping" << std::endl;
            return;
        }

        // Runtime validation: if we have recorded array-layer metadata for this
        // image, ensure the requested baseArrayLayer+layerCount is within bounds.
        if (image != VK_NULL_HANDLE) {
            auto layersOpt = resources.getImageArrayLayers(image);
            if (layersOpt.has_value()) {
                uint32_t recordedLayers = layersOpt.value();
                // Detect obvious out-of-range requests
                if (baseArrayLayer >= recordedLayers || baseArrayLayer + layerCount > recordedLayers) {
                    auto entry = resources.find((uintptr_t)image);
                    std::string desc = entry ? entry->desc : std::string("(unknown)");
                    std::cerr << "[VulkanApp] ERROR: requested array layer range out-of-bounds for image=" << (void*)image
                              << " desc='" << desc << "' base=" << baseArrayLayer << " count=" << layerCount
                              << " recordedLayers=" << recordedLayers << std::endl;
                    throw std::runtime_error("recordTransitionImageLayoutLayer: requested array layer range out-of-bounds");
                }
            }
        }

        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = effectiveOld;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        auto aspectFromFormat = [](VkFormat fmt) -> VkImageAspectFlags {
            switch (fmt) {
                case VK_FORMAT_D16_UNORM:
                case VK_FORMAT_X8_D24_UNORM_PACK32:
                case VK_FORMAT_D32_SFLOAT:
                    return VK_IMAGE_ASPECT_DEPTH_BIT;
                case VK_FORMAT_D16_UNORM_S8_UINT:
                case VK_FORMAT_D24_UNORM_S8_UINT:
                case VK_FORMAT_D32_SFLOAT_S8_UINT:
                    return VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
                default:
                    return VK_IMAGE_ASPECT_COLOR_BIT;
            }
        };
        barrier.subresourceRange.aspectMask = aspectFromFormat(format);
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;

        VkPipelineStageFlags2 sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if ((effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            barrier.srcAccessMask = 0;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            // Use COMPUTE_SHADER_BIT instead of FRAGMENT_SHADER_BIT to remain
            // compatible with transfer-only queues that support compute
            // (VUID-06461). The inter-queue semaphore already provides the
            // execution dependency for fragment shader work.
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) {
            // Depth attachment → read-only (e.g. water pass reusing the same depth buffer)
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            // Depth read-only → attachment (restore write access)
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            // Compute shader wrote to image → prepare for mipmap generation blit
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // Compute shader wrote to image → ready for fragment shader sampling
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else {
            throw std::runtime_error(
                std::string("[recordTransitionImageLayoutLayer] Unhandled transition: old=") +
                std::to_string((int)effectiveOld) + " new=" + std::to_string((int)newLayout));
        }

        barrier.srcStageMask = sourceStage;
        barrier.dstStageMask = destinationStage;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
        // Record pending layout update for this command buffer so the
        // authoritative map is only updated when the command buffer actually
        // completes. This avoids marking the global tracked layout as changed
        // before the GPU has executed the recorded barrier (which would allow
        // other threads to skip required barriers prematurely).
        {
            std::lock_guard<std::mutex> plk(pendingLayoutMutex);
            VulkanApp::PendingLayoutUpdate up;
            up.image = image;
            up.newLayout = newLayout;
            up.baseArrayLayer = baseArrayLayer;
            up.layerCount = layerCount;
            up.isBarrier = true;
            commandBufferPendingLayouts[commandBuffer].push_back(up);
        }
    }

void VulkanApp::transitionImageLayoutLayer(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount) {
    if (image == VK_NULL_HANDLE) throw std::runtime_error("transitionImageLayoutLayer called with VK_NULL_HANDLE image");
    runSingleTimeCommands([&](VkCommandBuffer commandBuffer){
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        auto aspectFromFormat = [](VkFormat fmt) -> VkImageAspectFlags {
            switch (fmt) {
                case VK_FORMAT_D16_UNORM:
                case VK_FORMAT_X8_D24_UNORM_PACK32:
                case VK_FORMAT_D32_SFLOAT:
                    return VK_IMAGE_ASPECT_DEPTH_BIT;
                case VK_FORMAT_D16_UNORM_S8_UINT:
                case VK_FORMAT_D24_UNORM_S8_UINT:
                case VK_FORMAT_D32_SFLOAT_S8_UINT:
                    return VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
                default:
                    return VK_IMAGE_ASPECT_COLOR_BIT;
            }
        };
        barrier.subresourceRange.aspectMask = aspectFromFormat(format);
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;

        // If a caller requested TRANSFER_DST_OPTIMAL for a depth-format
        // image, that's invalid unless the image was created with
        // VK_IMAGE_USAGE_TRANSFER_DST_BIT. We cannot query usage flags
        // here, so defensively map depth-image TRANSFER_DST transitions
        // to SHADER_READ_ONLY_OPTIMAL to avoid emitting invalid barriers.
        // This preserves the intent of initializing to a shader-readable
        // state without requiring a transfer usage bit.
        VkImageAspectFlags aspectMask = barrier.subresourceRange.aspectMask;
        if ((aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            throw std::runtime_error("transitionImageLayoutLayer: TRANSFER_DST_OPTIMAL requested for depth image (no fallback allowed)");
        }

        // Determine authoritative oldLayout for these layers and prefer the
        // app-tracked layout if it disagrees with the caller-supplied value.
        VkImageLayout effectiveOld = oldLayout;
        {
            std::lock_guard<std::mutex> lk(imageLayoutMutex);
            uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)baseArrayLayer;
            auto it = imageLayerLayouts.find(key);
            if (it != imageLayerLayouts.end()) {
                VkImageLayout tracked = it->second;
                if (tracked != oldLayout) {
                    std::cerr << "[VulkanApp] transitionImageLayoutLayer: image=" << (void*)image << " callerOld=" << (int)oldLayout << " trackedOld=" << (int)tracked << " -> using tracked" << std::endl;
                    effectiveOld = tracked;
                }
            } else {
                imageLayerLayouts[key] = oldLayout;
            }
        }

        if (effectiveOld > 7 && effectiveOld < 1000000000u) {
            std::cerr << "[VulkanApp] WARNING: invalid effectiveOld " << effectiveOld
                      << " for image " << (void*)image << ", clamping to UNDEFINED" << std::endl;
            effectiveOld = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // If the authoritative (tracked) layout already equals the requested
        // final layout, there's nothing to emit.
        if (effectiveOld == newLayout) {
            return;
        }

        VkPipelineStageFlags2 sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        // Common transitions supported here — add cases as needed.
        if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if ((effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || effectiveOld == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            // Transition from shader-read back to a depth attachment layout (write or read-only).
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            // Initialize depth-format image into a depth attachment layout from UNDEFINED.
            barrier.srcAccessMask = 0;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        } else if (effectiveOld == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // Initialize any image (color or depth) directly to shader-read layout.
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else {
            throw std::runtime_error("transitionImageLayoutLayer: Unsupported image layout transition requested (no fallback allowed)");
        }

        barrier.srcStageMask = sourceStage;
        barrier.dstStageMask = destinationStage;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    });

        // After performing the synchronous transition, update the authoritative
        // tracked layout for the affected layers so future record-time callers
        // observe the correct oldLayout (keeps this helper consistent with
        // recordTransitionImageLayoutLayer which updates the tracked map).
        {
            std::lock_guard<std::mutex> lk(imageLayoutMutex);
            for (uint32_t l = 0; l < layerCount; ++l) {
                uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)(baseArrayLayer + l);
                imageLayerLayouts[key] = newLayout;
            }
        }
}

// Force a synchronous transition on the GPU for the specified image layers
// even if the app-tracked layout would normally suppress emitting a barrier.
void VulkanApp::transitionImageLayoutLayerForce(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t baseArrayLayer, uint32_t layerCount) {
    if (image == VK_NULL_HANDLE) throw std::runtime_error("transitionImageLayoutLayerForce called with VK_NULL_HANDLE image");
    runSingleTimeCommands([&](VkCommandBuffer commandBuffer){
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        auto aspectFromFormat = [](VkFormat fmt) -> VkImageAspectFlags {
            switch (fmt) {
                case VK_FORMAT_D16_UNORM:
                case VK_FORMAT_X8_D24_UNORM_PACK32:
                case VK_FORMAT_D32_SFLOAT:
                    return VK_IMAGE_ASPECT_DEPTH_BIT;
                case VK_FORMAT_D16_UNORM_S8_UINT:
                case VK_FORMAT_D24_UNORM_S8_UINT:
                case VK_FORMAT_D32_SFLOAT_S8_UINT:
                    return VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
                default:
                    return VK_IMAGE_ASPECT_COLOR_BIT;
            }
        };
        barrier.subresourceRange.aspectMask = aspectFromFormat(format);
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
        barrier.subresourceRange.layerCount = layerCount;

        VkPipelineStageFlags2 sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        // Choose access masks and stages similar to transitionImageLayoutLayer
        VkImageAspectFlags aspectMask = barrier.subresourceRange.aspectMask;
        if ((aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            throw std::runtime_error("transitionImageLayoutLayerForce: TRANSFER_DST_OPTIMAL requested for depth image (no fallback allowed)");
        }

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            // Force initial transition for depth images from UNDEFINED into a depth attachment layout.
            barrier.srcAccessMask = 0;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else if ((oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)) {
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            } else {
                barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            sourceStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            // Initialize any image (color or depth) directly to shader-read layout.
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        } else {
            throw std::runtime_error("transitionImageLayoutLayerForce: Unsupported image layout transition requested (no fallback allowed)");
        }

        barrier.srcStageMask = sourceStage;
        barrier.dstStageMask = destinationStage;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    });

    // Update authoritative tracked layout after forcing the transition
    {
        std::lock_guard<std::mutex> lk(imageLayoutMutex);
        for (uint32_t l = 0; l < layerCount; ++l) {
            uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)(baseArrayLayer + l);
            imageLayerLayouts[key] = newLayout;
        }
    }
}

// Update the authoritative tracked layout for the specified image layers
// without emitting any pipeline barrier. Useful when the layout change is
// performed implicitly by a render pass (finalLayout) or other recorded
// commands and we only need to synchronize our tracked state.
void VulkanApp::setImageLayoutTracked(VkImage image, VkImageLayout newLayout, uint32_t baseArrayLayer, uint32_t layerCount) {
    if (image == VK_NULL_HANDLE) return;
    std::lock_guard<std::mutex> lk(imageLayoutMutex);
    for (uint32_t l = 0; l < layerCount; ++l) {
        uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)(baseArrayLayer + l);
        imageLayerLayouts[key] = newLayout;
    }
}

VkImageLayout VulkanApp::getImageLayoutTracked(VkImage image, uint32_t baseArrayLayer) const {
    if (image == VK_NULL_HANDLE) return VK_IMAGE_LAYOUT_UNDEFINED;
    std::lock_guard<std::mutex> lk(imageLayoutMutex);
    uint64_t key = ((uint64_t)(uintptr_t)image << 32) | (uint64_t)baseArrayLayer;
    auto it = imageLayerLayouts.find(key);
    if (it != imageLayerLayouts.end()) return it->second;
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanApp::recordTrackedLayoutForCommandBuffer(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout newLayout, uint32_t baseArrayLayer, uint32_t layerCount) {
    if (image == VK_NULL_HANDLE) return;
    // If no command buffer was provided, update authoritative map immediately
    if (commandBuffer == VK_NULL_HANDLE) {
        setImageLayoutTracked(image, newLayout, baseArrayLayer, layerCount);
        return;
    }

    std::lock_guard<std::mutex> plk(pendingLayoutMutex);
    VulkanApp::PendingLayoutUpdate up;
    up.image = image;
    up.newLayout = newLayout;
    up.baseArrayLayer = baseArrayLayer;
    up.layerCount = layerCount;
    up.isBarrier = false;
    commandBufferPendingLayouts[commandBuffer].push_back(up);
    {
        auto entry = resources.find((uintptr_t)image);
        std::string desc = entry ? entry->desc : std::string("(unknown)");
#if 0
        std::cerr << "[VulkanApp] recordTrackedLayoutForCommandBuffer: cmd=" << (void*)commandBuffer
                  << " image=" << (void*)image
                  << " desc=" << desc
                  << " new=" << layoutName(newLayout)
                  << " baseLayer=" << baseArrayLayer
                  << " layerCount=" << layerCount
                  << " isBarrier=0" << std::endl;
#endif
    }
}

// Deferred-destruction helpers
void VulkanApp::deferDestroyUntilAllPending(std::function<void()> destroyFn) {
    // Always enqueue a VK_NULL_HANDLE deferred destroy which means "wait
    // for all pending command buffers AND for in-flight frame fences to
    // signal". Running the callback immediately when no transfer command
    // buffers are outstanding risks destroying resources that are still
    // referenced by submitted render command buffers; defer to the
    // centralized processor to make the destruction decision.
    std::lock_guard<std::mutex> dd(deferredDestroyMutex);
    deferredDestroys.emplace_back(VK_NULL_HANDLE, destroyFn);
    std::cerr << "[VulkanApp] deferDestroyUntilAllPending: scheduled wait-for-all destroy (queue size=" << deferredDestroys.size() << ")" << std::endl;
}

void VulkanApp::deferDestroyUntilFence(VkFence fence, std::function<void()> destroyFn) {
    if (fence == VK_NULL_HANDLE) {
        deferDestroyUntilAllPending(destroyFn);
        return;
    }
    std::lock_guard<std::mutex> dd(deferredDestroyMutex);
    deferredDestroys.emplace_back(fence, destroyFn);
}

bool VulkanApp::hasPendingCommandBuffers() {
    std::lock_guard<std::mutex> lk(pendingCmdMutex);
    return !pendingCommandBuffers.empty();
}

void VulkanApp::addExtraWaitSemaphore(VkSemaphore sem, VkPipelineStageFlags stage) {
    std::lock_guard<std::mutex> lk(pendingCmdMutex);
    extraWaitSemaphores.emplace_back(sem, stage);
}



void VulkanApp::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    // Perform buffer->image copy on the transfer queue to avoid blocking graphics.
    runSingleTimeCommandsOnTransfer([&](VkCommandBuffer commandBuffer){
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    });
}

std::vector<VkCommandBuffer> VulkanApp::createCommandBuffers() {
    commandBuffers.clear();
    commandBuffers.resize(swapchainImages.size());
    // Allocate each command buffer from its own per-frame pool so that
    // vkResetCommandPool only resets the buffer for the current frame.
    for (uint32_t i = 0; i < commandBuffers.size(); ++i) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = frameCommandPools[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    return commandBuffers;
}

void VulkanApp::createSyncObjects() {
    // Create semaphores per frame-in-flight to avoid reuse before presentation completes.
    // Fences are per frame-in-flight for CPU-GPU synchronization.
    const uint32_t MAX_FRAMES_IN_FLIGHT = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    const uint32_t numImages = static_cast<uint32_t>(swapchainImages.size());

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // imageAvailableSemaphores: one per CPU frame-in-flight slot, used to wait on acquire
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    // renderFinishedSemaphores: one per swapchain image to avoid re-signaling a semaphore
    // that is still in use by the presentation engine (VUID-vkQueueSubmit-pSignalSemaphores-00067).
    renderFinishedSemaphores.resize(numImages);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create imageAvailableSemaphore for frame " + std::to_string(i));
        }
        resources.addSemaphore(imageAvailableSemaphores[i], "VulkanApp: imageAvailableSemaphore");
    }
    for (uint32_t i = 0; i < numImages; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create renderFinishedSemaphore for image " + std::to_string(i));
        }
        resources.addSemaphore(renderFinishedSemaphores[i], "VulkanApp: renderFinishedSemaphore");
    }
    
    // Fences per frame-in-flight
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence for frame " + std::to_string(i));
        }
        // Register fence
        resources.addFence(inFlightFences[i], "VulkanApp: inFlightFence");
    }

    // imagesInFlight tracks which fence is using each swapchain image (initialized null)
    imagesInFlight.clear();
    imagesInFlight.resize(numImages, VK_NULL_HANDLE);

    // Create the upload timeline semaphore (replaces per-upload binary semaphores)
    {
        VkSemaphoreTypeCreateInfo typeInfo{};
        typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue = 1; // start at 1 to avoid RADV value-0 wait crash

        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        createInfo.pNext = &typeInfo;
        if (vkCreateSemaphore(device, &createInfo, nullptr, &uploadTimeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create upload timeline semaphore");
        }
        resources.addSemaphore(uploadTimeline, "VulkanApp: uploadTimeline");
        uploadTimelineValue.store(1); // start at 1 (initialValue=1 avoids RADV value-0 crash)
    }

    // async submission bookkeeping
    pendingCommandBuffers.clear();
    semaphoresPendingDestroy.clear();
    extraWaitSemaphores.clear();
}

TextureImage VulkanApp::createTextureImageArray(const std::vector<std::string>& filenames, bool srgb) {
    TextureImage textureImage;
    if (filenames.empty()) throw std::runtime_error("createTextureImageArray: empty filename list");

    int texWidth = 0, texHeight = 0, texChannels = 0;
    std::vector<unsigned char*> layersData;
    layersData.reserve(filenames.size());

    for (size_t i = 0; i < filenames.size(); ++i) {
        unsigned char* pixels = stbi_load(filenames[i].c_str(), &texWidth, &texHeight, &texChannels, 4);
        if (!pixels) {
            // free any previously loaded
            for (auto p : layersData) if (p) stbi_image_free(p);
            throw std::runtime_error(std::string("failed to load texture image: ") + filenames[i]);
        }
        layersData.push_back(pixels);
    }

    const uint32_t layerCount = static_cast<uint32_t>(layersData.size());
    VkDeviceSize layerSize = texWidth * texHeight * 4;
    VkDeviceSize imageSize = layerSize * layerCount;

    // If the caller requested sRGB handling, convert loaded sRGB data to linear before storing as UNORM
    if (srgb) {
        for (uint32_t i = 0; i < layerCount; ++i) {
            convertSRGB8ToLinearInPlace(layersData[i], static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight));
        }
    }

    // create staging buffer containing all layers consecutively
    Buffer stagingBuffer = createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    for (uint32_t i = 0; i < layerCount; ++i) {
        memcpy((unsigned char*)stagingBuffer.mappedData + layerSize * i, layersData[i], layerSize);
    }

    for (auto p : layersData) stbi_image_free(p);

    textureImage.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    // choose format: use UNORM for array textures
    VkFormat chosenFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // create image with arrayLayers = layerCount
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = 0;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = textureImage.mipLevels;
    imageInfo.arrayLayers = layerCount;
    imageInfo.format = chosenFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // need transfer src/dst for mipmap generation and sampled for shader access
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VmaAllocationInfo allocInfo;
    VmaAllocation allocation;
    if (vmaCreateImage(vma.allocator, &imageInfo, &allocCI, &textureImage.image, &allocation, &allocInfo) != VK_SUCCESS) {
        stagingBuffer.buffer = VK_NULL_HANDLE;
        stagingBuffer.memory = VK_NULL_HANDLE;
        throw std::runtime_error("failed to create texture array image with VMA!");
    }
    textureImage.allocation = allocation;
    textureImage.memory = allocInfo.deviceMemory;
    resources.addImageVma(textureImage.image, allocation, "VulkanApp: textureArrayImage");
    printf("[VulkanApp] createImage(array): image=%p layers=%u mipLevels=%u format=%d\n", (void*)textureImage.image, (unsigned)imageInfo.arrayLayers, textureImage.mipLevels, (int)chosenFormat);

    // copy buffer to image per-layer using the app helpers so tracked
    // layouts and pending updates are recorded and applied correctly.
    runSingleTimeCommands([&](VkCommandBuffer commandBuffer){
        // Transition entire image to TRANSFER_DST_OPTIMAL (records pending update)
        recordTransitionImageLayoutLayer(commandBuffer,
                                         textureImage.image,
                                         chosenFormat,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         textureImage.mipLevels,
                                         0,
                                         layerCount);

        std::vector<VkBufferImageCopy> regions(layerCount);
        for (uint32_t i = 0; i < layerCount; ++i) {
            VkBufferImageCopy region{};
            region.bufferOffset = layerSize * i;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = i;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0,0,0};
            region.imageExtent = { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1 };
            regions[i] = region;
        }

        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.buffer, textureImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(regions.size()), regions.data());

        // Transition to SHADER_READ_ONLY_OPTIMAL (records pending update)
        recordTransitionImageLayoutLayer(commandBuffer,
                                         textureImage.image,
                                         chosenFormat,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                         textureImage.mipLevels,
                                         0,
                                         layerCount);
    });

    // generate mipmaps for the array texture (per-layer)
    generateMipmaps(textureImage.image, chosenFormat, texWidth, texHeight, textureImage.mipLevels, layerCount);

    // Transfer completed synchronously; destroy staging resources now.
    destroyBuffer(stagingBuffer);

    // create view for array texture
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = chosenFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = textureImage.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;

    if (vkCreateImageView(device, &viewInfo, nullptr, &textureImage.view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view (array)!");
    }
    printf("[VulkanApp] createImageView(array): view=%p image=%p\n", (void*)textureImage.view, (void*)textureImage.image);
    // Register image view for final-sweep safety
    resources.addImageView(textureImage.view, "VulkanApp::createTextureImageArray view");

    return textureImage;
}

void VulkanApp::createTextureImageView(TextureImage &textureImage) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = textureImage.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    // match the image format (use UNORM view)
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = textureImage.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &textureImage.view) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }
    printf("[VulkanApp] createImageView: view=%p image=%p\n", (void*)textureImage.view, (void*)textureImage.image);
    // Register image view
    resources.addImageView(textureImage.view, "VulkanApp: textureImage.view");
}

VkSampler VulkanApp::createTextureSampler(uint32_t mipLevels) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // enable anisotropic filtering when supported by the device
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    if (deviceProperties.limits.maxSamplerAnisotropy > 1.0f) {
        samplerInfo.anisotropyEnable = VK_TRUE;
        // clamp requested anisotropy to device maximum
        float desiredAniso = std::min<float>(16.0f, deviceProperties.limits.maxSamplerAnisotropy);
        samplerInfo.maxAnisotropy = desiredAniso;
    } else {
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
    }
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)mipLevels;
    samplerInfo.mipLodBias = 0.0f;
    VkSampler textureSampler = VK_NULL_HANDLE;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
    printf("[VulkanApp] createSampler: sampler=%p mipLevels=%u\n", (void*)textureSampler, (unsigned)mipLevels);
    // Register sampler in resource registry for final-sweep safety
    resources.addSampler(textureSampler, "VulkanApp: textureSampler");
    return textureSampler;
}

void VulkanApp::createDescriptorSetLayout() {
    // binding 0 : uniform buffer (vertex shader)
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    // UBO is referenced by vertex, fragment, tessellation, and geometry stages
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT;

    // bindings 1..3: arrays of combined image samplers (albedo / normal / height)
    // bindings 1..3: one combined image sampler each (we use a texture2D array as the image view)
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    VkDescriptorSetLayoutBinding normalSamplerBinding{};
    normalSamplerBinding.binding = 2;
    normalSamplerBinding.descriptorCount = 1;
    normalSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    normalSamplerBinding.pImmutableSamplers = nullptr;
    normalSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding heightSamplerBinding{};
    heightSamplerBinding.binding = 3;
    heightSamplerBinding.descriptorCount = 1;
    heightSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    heightSamplerBinding.pImmutableSamplers = nullptr;
    // Height sampler is used by fragment shader and tessellation evaluation shader (for displacement)
    heightSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // binding 4: shadow map sampler
    VkDescriptorSetLayoutBinding shadowSamplerBinding{};
    shadowSamplerBinding.binding = 4;
    shadowSamplerBinding.descriptorCount = 1;
    shadowSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowSamplerBinding.pImmutableSamplers = nullptr;
    shadowSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 6: Sky UBO
    VkDescriptorSetLayoutBinding skyBinding{};
    skyBinding.binding = 6;
    skyBinding.descriptorCount = 1;
    skyBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    skyBinding.pImmutableSamplers = nullptr;
    skyBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    // binding 7: Water params SSBO (for water shader) - use storage buffer like Materials
    VkDescriptorSetLayoutBinding waterParamsBinding{};
    waterParamsBinding.binding = 7;
    waterParamsBinding.descriptorCount = 1;
    waterParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    waterParamsBinding.pImmutableSamplers = nullptr;
    // Make the water params visible to fragment, tessellation evaluation, and tessellation control shaders
    waterParamsBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    // Per-instance / per-draw descriptor set uses bindings: 0 (UBO), 1..3 (samplers), 4 (shadow cascade 0),
    // 5 (Materials SSBO), 6 (Sky UBO), 7 (water params), 8 (shadow cascade 1), 9 (shadow cascade 2)
    // Note: Materials (binding 5) is declared in shaders as set=0 binding=5, so include it in the main layout.

    // binding 8: shadow map cascade 1
    VkDescriptorSetLayoutBinding shadowCascade1Binding{};
    shadowCascade1Binding.binding = 8;
    shadowCascade1Binding.descriptorCount = 1;
    shadowCascade1Binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowCascade1Binding.pImmutableSamplers = nullptr;
    shadowCascade1Binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 9: shadow map cascade 2
    VkDescriptorSetLayoutBinding shadowCascade2Binding{};
    shadowCascade2Binding.binding = 9;
    shadowCascade2Binding.descriptorCount = 1;
    shadowCascade2Binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowCascade2Binding.pImmutableSamplers = nullptr;
    shadowCascade2Binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 10: Water render UBO (time parameter for water shaders)
    VkDescriptorSetLayoutBinding waterRenderUBOBinding{};
    waterRenderUBOBinding.binding = 10;
    waterRenderUBOBinding.descriptorCount = 1;
    waterRenderUBOBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    waterRenderUBOBinding.pImmutableSamplers = nullptr;
    waterRenderUBOBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;

    // binding 11: 360° environment cubemap sampler for solid-shader reflections
    VkDescriptorSetLayoutBinding envMapBinding{};
    envMapBinding.binding = 11;
    envMapBinding.descriptorCount = 1;
    envMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    envMapBinding.pImmutableSamplers = nullptr;
    envMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 12: roughness map array
    VkDescriptorSetLayoutBinding roughnessSamplerBinding{};
    roughnessSamplerBinding.binding = 12;
    roughnessSamplerBinding.descriptorCount = 1;
    roughnessSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    roughnessSamplerBinding.pImmutableSamplers = nullptr;
    roughnessSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 13: ambient occlusion map array
    VkDescriptorSetLayoutBinding aoSamplerBinding{};
    aoSamplerBinding.binding = 13;
    aoSamplerBinding.descriptorCount = 1;
    aoSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    aoSamplerBinding.pImmutableSamplers = nullptr;
    aoSamplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 14> bindings = {
        uboLayoutBinding, samplerLayoutBinding, normalSamplerBinding, heightSamplerBinding,
        shadowSamplerBinding, /* material */ VkDescriptorSetLayoutBinding{}, skyBinding,
        waterParamsBinding, shadowCascade1Binding, shadowCascade2Binding, waterRenderUBOBinding,
        envMapBinding, roughnessSamplerBinding, aoSamplerBinding
    };
    // Fill the material binding at position 5
    bindings[5].binding = 5;
    bindings[5].descriptorCount = 1;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].pImmutableSamplers = nullptr;
    bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    // Binding flags — enable update-after-bind for binding 11 (cubemap environment map)
    // so that vkUpdateDescriptorSets can write binding 11 while a command buffer
    // referencing this descriptor set is still pending (the cubemap render path
    // swaps between a dummy cubemap and the real one every frame).
    std::array<VkDescriptorBindingFlags, 14> bindingFlags{};
    bindingFlags.fill(0);
    bindingFlags[11] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &bindingFlagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Register the main descriptor set layout for inspection/cleanup
    resources.addDescriptorSetLayout(descriptorSetLayout, "VulkanApp: descriptorSetLayout");

    // Allocate the main UBO/sampler/materials descriptor sets (one per frame)
    const uint32_t MAIN_DESC_SETS = MAX_FRAMES_IN_FLIGHT;
    mainDescriptorSets.clear();
    mainDescriptorSets.resize(MAIN_DESC_SETS);
    for (uint32_t i = 0; i < MAIN_DESC_SETS; ++i) {
        mainDescriptorSets[i] = createDescriptorSet(descriptorSetLayout);
    }

    // Create a separate material descriptor layout used for materials only
    std::array<VkDescriptorSetLayoutBinding, 1> materialBindings = { bindings[5] };
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
    materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    materialLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
    materialLayoutInfo.pBindings = materialBindings.data();

    if (vkCreateDescriptorSetLayout(device, &materialLayoutInfo, nullptr, &materialDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create material descriptor set layout!");
    }

    // Register material descriptor set layout
    resources.addDescriptorSetLayout(materialDescriptorSetLayout, "VulkanApp::createDescriptorSetLayout materialDescriptorSetLayout");

    // If we later add a normal map sampler (binding 2), extend bindings dynamically when required by the app.
}

void VulkanApp::createDepthResources() {
    // simple depth resources using a 32-bit float depth format
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapchainExtent.width;
    imageInfo.extent.height = swapchainExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VmaAllocationInfo allocInfo;
    if (vmaCreateImage(vma.allocator, &imageInfo, &allocCI, &depthImage, &depthImageAllocation, &allocInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image with VMA!");
    }
    depthImageMemory = allocInfo.deviceMemory;
    resources.addImageVma(depthImage, depthImageAllocation, "VulkanApp: depthImage");

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image view!");
    }

    resources.addImageView(depthImageView, "VulkanApp: depthImageView");


    // transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    runSingleTimeCommands([&](VkCommandBuffer commandBuffer){
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = depthImage;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    });
    // Update authoritative tracked layout so the app's layout map reflects
    // the synchronous transition we just performed. This prevents later
    // record-time callers from assuming an UNDEFINED layout when the GPU
    // already has the image in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    setImageLayoutTracked(depthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
}

void VulkanApp::createDescriptorPool(uint32_t uboCount, uint32_t samplerCount) {
    // Reserve descriptors: uniform buffers, combined image samplers, and storage buffers for materials
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Each descriptor set will reference the per-set scene UBO (binding 0)
    // and the shared Sky UBO (binding 6). Reserve two uniform descriptors per set.
    poolSizes[0].descriptorCount = uboCount * 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = samplerCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // Increase storage buffer descriptors for compute workloads (was: uboCount)
    poolSizes[2].descriptorCount = uboCount * 8;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    // Allow freeing individual descriptor sets (vegetation, etc.) and support UPDATE_AFTER_BIND layouts
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    // Increase maxSets to support many compute/graphics allocations
    poolInfo.maxSets = uboCount * 16;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
    // Register default descriptor pool for app-managed allocations
    resources.addDescriptorPool(descriptorPool, "VulkanApp: descriptorPool");
}

VkDescriptorSet VulkanApp::createDescriptorSet(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    if (layout == VK_NULL_HANDLE) {
        std::cerr << "[VulkanApp::createDescriptorSet] ERROR: requested layout is VK_NULL_HANDLE" << std::endl;
        throw std::runtime_error("createDescriptorSet called with VK_NULL_HANDLE layout");
    }

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    {
        // Serialize allocations from the shared app descriptor pool to avoid
        // driver races when multiple threads allocate descriptor sets concurrently.
        std::lock_guard<std::mutex> lk(descriptorAllocMutex);
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            std::cerr << "[VulkanApp::createDescriptorSet] vkAllocateDescriptorSets failed for layout=" << (void*)layout << std::endl;
            throw std::runtime_error("failed to allocate descriptor set!");
        }
    }
    // Register descriptor set so manager can track it for inspection
    resources.addDescriptorSet(descriptorSet, "VulkanApp: descriptorSet");
    if ((uint64_t)descriptorSet == 0x6c100000006c1ULL) {
        std::cerr << "[VulkanApp::createDescriptorSet] *** CRITICAL: allocated suspicious handle 0x6c100000006c1 ***" << std::endl;
    }
    std::cout << "[VulkanApp::createDescriptorSet] allocated descriptorSet=" << (void*)descriptorSet << " layout=" << (void*)layout << std::endl;
    return descriptorSet;
}

VkResult VulkanApp::allocateDescriptorSetsThreadSafe(const VkDescriptorSetAllocateInfo* pAllocInfo, VkDescriptorSet* pDescriptorSets) {
    std::lock_guard<std::mutex> lk(descriptorAllocMutex);
    VkResult res = vkAllocateDescriptorSets(device, pAllocInfo, pDescriptorSets);
    if (res != VK_SUCCESS) {
        std::cerr << "[VulkanApp::allocateDescriptorSetsThreadSafe] vkAllocateDescriptorSets failed: " << res << std::endl;
    } else {
        for (uint32_t i = 0; i < pAllocInfo->descriptorSetCount; ++i) {
            std::cerr << "[RAW ALLOC] allocateDescriptorSetsThreadSafe: descSet=" << (void*)pDescriptorSets[i] << " pool=" << (void*)pAllocInfo->descriptorPool << " count=" << pAllocInfo->descriptorSetCount << std::endl;
            if ((uint64_t)pDescriptorSets[i] == 0x6c100000006c1ULL) {
                std::cerr << "[VulkanApp::allocateDescriptorSetsThreadSafe] *** ALLOCATED BAD HANDLE 0x6c100000006c1 ***" << std::endl;
            }
        }
    }
    return res;
}

VkDescriptorSet VulkanApp::createMaterialDescriptorSet() {
    if (materialDescriptorSetLayout == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    return createDescriptorSet(materialDescriptorSetLayout);
}

void VulkanApp::updateDescriptorSet(const std::vector<VkWriteDescriptorSet> &descriptors) {
    // Filter out any writes that would pass VK_NULL_HANDLE for buffer or image
    // (some callers may attempt to update before resources are allocated).
    std::vector<VkWriteDescriptorSet> filtered;
    filtered.reserve(descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const VkWriteDescriptorSet &w = descriptors[i];
        if (w.pImageInfo) {
            if (w.pImageInfo[0].imageView == VK_NULL_HANDLE || w.pImageInfo[0].sampler == VK_NULL_HANDLE) {
#if 0
                std::cerr << "[VulkanApp::updateDescriptorSet] Skipping write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " due to null imageView/sampler" << std::endl;
#endif
                continue;
            }
#if 0
            std::cerr << "[VulkanApp::updateDescriptorSet] write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " type=" << w.descriptorType << " imageView=" << (void*)w.pImageInfo[0].imageView << " sampler=" << (void*)w.pImageInfo[0].sampler << std::endl;
#endif
            filtered.push_back(w);
        } else if (w.pBufferInfo) {
            if (w.pBufferInfo[0].buffer == VK_NULL_HANDLE) {
#if 0
                std::cerr << "[VulkanApp::updateDescriptorSet] Skipping write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " due to null buffer" << std::endl;
#endif
                continue;
            }
#if 0
            std::cerr << "[VulkanApp::updateDescriptorSet] write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " type=" << w.descriptorType << " buffer=" << (void*)w.pBufferInfo[0].buffer << " offset=" << w.pBufferInfo[0].offset << " range=" << w.pBufferInfo[0].range << std::endl;
#endif
            filtered.push_back(w);
        } else {
            // No image or buffer info; include as-is
            filtered.push_back(w);
        }
    }
    if (filtered.empty()) return;
    // Filter out writes targeting invalid descriptor set handles to prevent
    // VUID-VkWriteDescriptorSet-dstSet-00320 validation errors.
    std::vector<VkWriteDescriptorSet> safe;
    safe.reserve(filtered.size());
    for (auto &f : filtered) {
        if ((uint64_t)f.dstSet == 0x6c100000006c1ULL) {
            std::cerr << "[VulkanApp::updateDescriptorSet] *** SKIPPING BAD HANDLE 0x6c100000006c1 *** binding=" << f.dstBinding << std::endl;
            continue;
        }
        safe.push_back(f);
    }
    if (safe.empty()) return;
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(safe.size()), safe.data(), 0, nullptr);
}

void VulkanApp::updateDescriptorSet(std::initializer_list<VkWriteDescriptorSet> descriptors) {
    std::vector<VkWriteDescriptorSet> descriptorWrites(descriptors);
    std::vector<VkWriteDescriptorSet> filtered;
    filtered.reserve(descriptorWrites.size());
    for (size_t i = 0; i < descriptorWrites.size(); ++i) {
        const VkWriteDescriptorSet &w = descriptorWrites[i];
        if (w.pImageInfo) {
            if (w.pImageInfo[0].imageView == VK_NULL_HANDLE || w.pImageInfo[0].sampler == VK_NULL_HANDLE) {
#if 0
                std::cerr << "[VulkanApp::updateDescriptorSet] Skipping init write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " due to null imageView/sampler" << std::endl;
#endif
                continue;
            }
            filtered.push_back(w);
        } else if (w.pBufferInfo) {
            if (w.pBufferInfo[0].buffer == VK_NULL_HANDLE) {
#if 0
                std::cerr << "[VulkanApp::updateDescriptorSet] Skipping init write[" << i << "] dstSet=" << (void*)w.dstSet << " binding=" << w.dstBinding << " due to null buffer" << std::endl;
#endif
                continue;
            }
            filtered.push_back(w);
        } else {
            filtered.push_back(w);
        }
    }
    if (filtered.empty()) return;
    std::vector<VkWriteDescriptorSet> safe;
    safe.reserve(filtered.size());
    for (auto &f : filtered) {
        if ((uint64_t)f.dstSet == 0x6c100000006c1ULL) {
            std::cerr << "[VulkanApp::updateDescriptorSet(init)] *** SKIPPING BAD HANDLE 0x6c100000006c1 *** binding=" << f.dstBinding << std::endl;
            continue;
        }
        safe.push_back(f);
    }
    if (safe.empty()) return;
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(safe.size()), safe.data(), 0, nullptr);
}


VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    // Track shader module creation
    resources.addShaderModule(shaderModule, "VulkanApp: shaderModule");
    return shaderModule;
}

VkDevice VulkanApp::getDevice() const {
    return device;
}

VkPipelineLayout VulkanApp::getPipelineLayout() const {
    return pipelineLayout;
}

std::pair<VkPipeline, VkPipelineLayout> VulkanApp::createGraphicsPipeline(
    std::initializer_list<VkPipelineShaderStageCreateInfo> stages,
    const std::vector<VkVertexInputBindingDescription>& bindingDescriptions,
    const std::vector<VkVertexInputAttributeDescription>& descriptions,
    const std::vector<VkDescriptorSetLayout>& setLayouts,
    const VkPushConstantRange* pushConstantRange,
    VkPolygonMode polygonMode,
    VkCullModeFlagBits cullMode,
    bool depthWrite,
    bool colorWrite,
    VkCompareOp depthCompare,
    VkPrimitiveTopology topology,
    bool depthClampEnable,
    const std::vector<VkFormat>& colorFormats,
    VkFormat depthFormat,
    bool noColorAttachment,
    bool depthBiasEnable,
    VkRenderPass legacyRenderPass) {

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages(stages);
    const std::vector<VkVertexInputAttributeDescription>& attributeDescriptions = descriptions;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swapchainExtent.width;
    viewport.height = (float) swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0,0};
    scissor.extent = swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = nullptr;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = depthClampEnable ? VK_TRUE : VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = polygonMode;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cullMode;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = depthBiasEnable ? VK_TRUE : VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    if (colorWrite) {
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    } else {
        colorBlendAttachment.colorWriteMask = 0;
    }
    colorBlendAttachment.blendEnable = VK_FALSE;

    // Determine effective color formats for dynamic rendering
    std::vector<VkFormat> effectiveColorFormats;
    if (noColorAttachment) {
        // depth-only pipeline: no color attachments
    } else if (colorFormats.empty()) {
        effectiveColorFormats = { swapchainImageFormat };
    } else {
        effectiveColorFormats = colorFormats;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = static_cast<uint32_t>(effectiveColorFormats.size());
    colorBlending.pAttachments = effectiveColorFormats.empty() ? nullptr : &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.empty() ? nullptr : setLayouts.data();
    // Diagnostic: print provided set layouts for easier debugging of descriptor mismatches
    if (setLayouts.empty()) {
        std::cerr << "[createGraphicsPipeline] setLayouts count=0\n";
    }
    if (pushConstantRange) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = pushConstantRange;
    } else {
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
    // Track pipeline layout for cleanup
    resources.addPipelineLayout(pipelineLayout, "VulkanApp: pipelineLayout");

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = depthCompare;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // If tessellation shaders are present, set input assembly topology to patch list and add tessellation state
    bool hasTessellation = false;
    for (const auto &s : shaderStages) {
        if (s.stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) {
            hasTessellation = true;
            break;
        }
    }
    VkPipelineTessellationStateCreateInfo tessState{};
    if (hasTessellation) {
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        tessState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
        tessState.patchControlPoints = 3; // triangles
    }

    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.pNext = nullptr;
    pipelineRenderingInfo.colorAttachmentCount = static_cast<uint32_t>(effectiveColorFormats.size());
    pipelineRenderingInfo.pColorAttachmentFormats = effectiveColorFormats.empty() ? nullptr : effectiveColorFormats.data();
    pipelineRenderingInfo.depthAttachmentFormat = depthFormat;
    pipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.subpass = 0;
    if (hasTessellation) pipelineInfo.pTessellationState = &tessState;

    if (legacyRenderPass != VK_NULL_HANDLE) {
        pipelineInfo.pNext = nullptr;
        pipelineInfo.renderPass = legacyRenderPass;
    } else {
        pipelineInfo.pNext = &pipelineRenderingInfo;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
    }
    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    // Track pipeline for cleanup
    resources.addPipeline(graphicsPipeline, "VulkanApp: graphicsPipeline");
    std::cout << "graphics pipeline created\n";
    registeredPipelines.push_back(graphicsPipeline);
    return {graphicsPipeline, pipelineLayout};
}

uint32_t VulkanApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

Buffer VulkanApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    Buffer buffer;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }
    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VmaAllocation allocation;
    VmaAllocationInfo allocInfo;
    if (vmaCreateBuffer(vma.allocator, &bufferInfo, &allocCI, &buffer.buffer, &allocation, &allocInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer with VMA!");

    buffer.allocation = allocation;
    buffer.mappedData = allocInfo.pMappedData;
    buffer.memory = allocInfo.deviceMemory;

    resources.addBufferVma(buffer.buffer, allocation, "VulkanApp: buffer.buffer");
    return buffer;
}



void VulkanApp::destroyBuffer(Buffer& buf) {
    if (buf.allocation && vma.allocator) {
        resources.removeBuffer(buf.buffer);
        vmaDestroyBuffer(vma.allocator, buf.buffer, buf.allocation);
    } else {
        if (buf.buffer != VK_NULL_HANDLE) {
            resources.removeBuffer(buf.buffer);
            vkDestroyBuffer(device, buf.buffer, nullptr);
        }
        if (buf.memory != VK_NULL_HANDLE) {
            resources.removeDeviceMemory(buf.memory);
            vkFreeMemory(device, buf.memory, nullptr);
        }
    }
    buf = {};
}

void VulkanApp::updateUniformBuffer(Buffer &uniform, void *data, size_t dataSize) {
    // Defensive: ensure memory is valid before mapping
    memcpy(uniform.mappedData, data, dataSize);
}

Buffer VulkanApp::createVertexBuffer(const std::vector<Vertex> &vertices) {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.mappedData, vertices.data(), (size_t)bufferSize);
    
    // Create device-local vertex buffer for fast GPU access
    Buffer vertexBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer on transfer queue
    runSingleTimeCommandsOnTransfer([&](VkCommandBuffer cmd){
        VkBufferCopy copyRegion{};
        copyRegion.size = bufferSize;
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, vertexBuffer.buffer, 1, &copyRegion);
    });
    
    // Transfer completed synchronously; destroy staging resources now.
    destroyBuffer(stagingBuffer);
    
    return vertexBuffer;
}

bool VulkanApp::isResourceRegistered(uintptr_t handle) const {
    if (handle == 0) return false;
    auto e = resources.find(handle);
    return e.has_value();
}

std::vector<VulkanApp::MemoryHeapBudget> VulkanApp::getMemoryBudgets() const {
    std::vector<MemoryHeapBudget> result;
    if (!physicalDevice) return result;

    // Chain VkPhysicalDeviceMemoryBudgetPropertiesEXT into VkPhysicalDeviceMemoryProperties2
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps{};
    budgetProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budgetProps;

    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProps2);

    for (uint32_t i = 0; i < memProps2.memoryProperties.memoryHeapCount; ++i) {
        MemoryHeapBudget hb{};
        hb.flags  = memProps2.memoryProperties.memoryHeaps[i].flags;
        hb.size   = memProps2.memoryProperties.memoryHeaps[i].size;
        hb.usage  = budgetProps.heapUsage[i];
        hb.budget = budgetProps.heapBudget[i];
        result.push_back(hb);
    }
    return result;
}

Buffer VulkanApp::createIndexBuffer(const std::vector<uint> &indices) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.mappedData, indices.data(), (size_t)bufferSize);
    
    // Create device-local index buffer for fast GPU access
    Buffer indexBuffer = createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer on transfer queue
    runSingleTimeCommandsOnTransfer([&](VkCommandBuffer cmd){
        VkBufferCopy copyRegion{};
        copyRegion.size = bufferSize;
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, indexBuffer.buffer, 1, &copyRegion);
    });
    
    // Transfer completed synchronously; destroy staging resources now.
    destroyBuffer(stagingBuffer);
    
    return indexBuffer;
}

Buffer VulkanApp::createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    // Create staging buffer (host-visible) to transfer data to GPU
    Buffer stagingBuffer = createBuffer(size, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    memcpy(stagingBuffer.mappedData, data, (size_t)size);
    
    // Create device-local buffer for fast GPU access
    Buffer gpuBuffer = createBuffer(size, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    // Copy from staging to device-local buffer, then insert a pipeline barrier
    // so that subsequent shader/compute reads see the transferred data.
    runSingleTimeCommandsOnTransfer([&](VkCommandBuffer cmd){
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, gpuBuffer.buffer, 1, &copyRegion);

        VkBufferMemoryBarrier2 bufBarrier{};
        bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
        bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.buffer = gpuBuffer.buffer;
        bufBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &bufBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    });
    
    // Transfer completed synchronously; destroy staging resources now.
    destroyBuffer(stagingBuffer);
    
    return gpuBuffer;
}

Buffer VulkanApp::createDeviceLocalBufferExclusive(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer stagingBuffer = createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    memcpy(stagingBuffer.mappedData, data, (size_t)size);

    Buffer gpuBuffer{};
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &gpuBuffer.buffer) != VK_SUCCESS)
        throw std::runtime_error("createDeviceLocalBufferExclusive: vkCreateBuffer failed");
    resources.addBuffer(gpuBuffer.buffer, "createDeviceLocalBufferExclusive: buffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, gpuBuffer.buffer, &memReq);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &gpuBuffer.memory) != VK_SUCCESS)
        throw std::runtime_error("createDeviceLocalBufferExclusive: vkAllocateMemory failed");
    vkBindBufferMemory(device, gpuBuffer.buffer, gpuBuffer.memory, 0);
    resources.addDeviceMemory(gpuBuffer.memory, "createDeviceLocalBufferExclusive: memory");

    runSingleTimeCommandsOnTransfer([&](VkCommandBuffer cmd){
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer.buffer, gpuBuffer.buffer, 1, &copyRegion);

        VkBufferMemoryBarrier2 bufBarrier{};
        bufBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufBarrier.buffer = gpuBuffer.buffer;
        bufBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &bufBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    });

    destroyBuffer(stagingBuffer);
    return gpuBuffer;
}

void VulkanApp::drawFrame() {
    const uint32_t MAX_FRAMES_IN_FLIGHT = static_cast<uint32_t>(inFlightFences.size());
    const uint32_t numImages = static_cast<uint32_t>(swapchainImages.size());
    uint32_t imageIndex;

    // Use the current CPU frame index as the semaphore index so the
    // acquire semaphore and submit wait semaphore are aligned per-frame.
    uint32_t semaphoreIndex = currentFrame;

    // Wait for the CPU frame fence for the current frame first. This limits
    // the number of outstanding acquired swapchain images to the number of
    // CPU frames-in-flight and ensures forward progress for vkAcquireNextImageKHR
    // when using an infinite timeout (UINT64_MAX).
    VkResult waitForFrameFence = vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    if (waitForFrameFence == VK_ERROR_DEVICE_LOST) return;
    if (waitForFrameFence != VK_SUCCESS) {
        std::cerr << "vkWaitForFences failed: " << waitForFrameFence << std::endl;
        return;
    }

    // Acquire next image using per-image semaphore
    VkResult r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[semaphoreIndex], VK_NULL_HANDLE, &imageIndex);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    } else if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        std::cerr << "vkAcquireNextImageKHR failed: " << r << std::endl;
        return;
    }

    // If a previous frame is using this image and it's a DIFFERENT frame than current,
    // we need to wait for it. With MAX_FRAMES_IN_FLIGHT >= swapchain images, this is rare.
    // Optimization: only wait if the image's fence differs from the one we just waited on.
    if (imagesInFlight.size() > imageIndex && 
        imagesInFlight[imageIndex] != VK_NULL_HANDLE &&
        imagesInFlight[imageIndex] != inFlightFences[currentFrame]) {
        VkResult res = vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        if (res == VK_ERROR_DEVICE_LOST) return;
        if (res != VK_SUCCESS) {
            std::cerr << "vkWaitForFences (image) failed: " << res << std::endl;
            return;
        }
    }

    // Mark this image as now being used by the current frame's fence
    if (imagesInFlight.size() > imageIndex) imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    // Record authoritative tracked layout for the acquired swapchain image.
    // The acquire operation yields the image in PRESENT_SRC_KHR for use
    // by the renderer. Populate the tracked map so preApply can see the
    // correct layout and avoid validation mismatches.
    {
        std::lock_guard<std::mutex> lk(imageLayoutMutex);
        uint64_t key = ((uint64_t)(uintptr_t)swapchainImages[imageIndex] << 32) | (uint64_t)0;
        imageLayerLayouts[key] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    // reset current frame fence so vkQueueSubmit can signal it
    VkResult resetFenceResult = vkResetFences(device, 1, &inFlightFences[currentFrame]);
    if (resetFenceResult == VK_ERROR_DEVICE_LOST) return;
    if (resetFenceResult != VK_SUCCESS) {
        std::cerr << "vkResetFences failed: " << resetFenceResult << std::endl;
        return;
    }
    // compute deltaTime for this frame
    double frameNow = glfwGetTime();
    float deltaTime = 0.0f;
    if (lastFrameTime > 0.0) deltaTime = static_cast<float>(frameNow - lastFrameTime);
    lastFrameTime = frameNow;
    if (!isLoading) {
        update(deltaTime);
    }

    // ImGui new frame (backend)
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (isLoading) {
        // Draw a centered "Loading..." text with white text on black background
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        ImVec2 display = ImGui::GetIO().DisplaySize;
        dl->AddRectFilled(ImVec2(0.0f, 0.0f), display, IM_COL32(0, 0, 0, 255));
        const char* text = "Loading...";
        ImVec2 textSize = ImGui::CalcTextSize(text);
        float x = (display.x - textSize.x) * 0.5f;
        float y = (display.y - textSize.y) * 0.5f;
        dl->AddText(ImVec2(x, y), IM_COL32(255, 255, 255, 255), text);
    } else {
        // call into the derived app to build the UI
        renderImGui();
    }

    // update FPS (simple moving average could be added)
    double now = glfwGetTime();
    if (imguiLastTime > 0.0) {
        double dt = now - imguiLastTime;
        if (dt > 0.0) imguiFps = static_cast<float>(1.0 / dt);
    }
    imguiLastTime = now;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Wait on the semaphore used for this acquire (indexed by semaphoreIndex)
    std::vector<VkSemaphore> waitSemaphoresVec = { imageAvailableSemaphores[semaphoreIndex] };
    std::vector<VkPipelineStageFlags> waitStagesVec = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    // Pull extra semaphores signaled by async generation submissions (if any)
    {
        std::lock_guard<std::mutex> lk(extraSemaphoreMutex);
        for (auto &e : extraWaitSemaphores) {
            waitSemaphoresVec.push_back(e.first);
            waitStagesVec.push_back(e.second);
        }
        // Move them to semaphoresPendingDestroy so they can be destroyed after this frame's fence signals
        if (!extraWaitSemaphores.empty()) {
            for (auto &e : extraWaitSemaphores) semaphoresPendingDestroy.emplace_back(e.first, inFlightFences[currentFrame]);
            extraWaitSemaphores.clear();
        }
    }

    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphoresVec.size());
    submitInfo.pWaitSemaphores = waitSemaphoresVec.data();
    submitInfo.pWaitDstStageMask = waitStagesVec.data();

    // Copy the wait arrays into local storage so address stays valid while submitInfo references them
    std::vector<VkSemaphore> localWaitSemaphores = std::move(waitSemaphoresVec);
    std::vector<VkPipelineStageFlags> localWaitStages = std::move(waitStagesVec);
    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(localWaitSemaphores.size());
    submitInfo.pWaitSemaphores = localWaitSemaphores.data();
    submitInfo.pWaitDstStageMask = localWaitStages.data();

    // Signal semaphore indexed by swapchain image to avoid re-signaling before
    // the presentation engine re-acquires the image (per-image semaphore).
    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // Debug: check commandBuffers size and imageIndex
    if (imageIndex >= commandBuffers.size()) {
        std::cerr << "[FATAL] imageIndex out of bounds for commandBuffers!" << std::endl;
        abort();
    }
    VkCommandBuffer &commandBuffer = commandBuffers[imageIndex];
    if (commandBuffer == VK_NULL_HANDLE) {
        std::cerr << "[FATAL] commandBuffer is VK_NULL_HANDLE!" << std::endl;
        abort();
    }
    // reset the per-frame command pool (and implicitly the command buffer) for this frame
    VkResult resetCmdResult;
    {
        // Acquire graphicsSubmitMutex first to match the lock ordering used by endSingleTimeCommands()
        std::lock_guard<std::mutex> lockQ(graphicsSubmitMutex);
        std::lock_guard<std::mutex> lockC(commandPoolMutex);
        resetCmdResult = vkResetCommandPool(device, frameCommandPools[imageIndex], 0);
    }
    if (resetCmdResult == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (resetCmdResult != VK_SUCCESS) {
        std::cerr << "vkResetCommandPool failed: " << resetCmdResult << std::endl;
        return;
    }

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginCmdResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (beginCmdResult != VK_SUCCESS) {
        std::cerr << "vkBeginCommandBuffer failed: " << beginCmdResult << std::endl;
        return;
    }

    // Process any completed async generation submissions and free their command buffers/fences/semaphores.
    // Must run BEFORE preRenderPass so newly-generated vegetation chunks are available for the shadow pass.
    processPendingCommandBuffers();

    // Only run scene hooks when the app is fully set up
    if (!isLoading) {
        // Hook for compute/barrier operations before render pass
        preRenderPass(commandBuffer);
    }

    // Transition swapchain image: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
    // Synchronization2: embed stage/access masks in the barrier struct.
    {
        VkImageMemoryBarrier2 colorBarrier{};
        colorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        colorBarrier.srcAccessMask = 0;
        colorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        colorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorBarrier.image = swapchainImages[imageIndex];
        colorBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &colorBarrier;

        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }

    VkClearValue clearColor = isLoading ? VkClearValue{{{0.0f, 0.0f, 0.0f, 1.0f}}}
                                        : VkClearValue{{{0.0f, 0.0f, 0.0f, 0.0f}}};
    VkClearValue clearDepth{};
    clearDepth.depthStencil = {1.0f, 0};

    VkRenderingAttachmentInfo colorAttachmentInfo{};
    colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachmentInfo.imageView = swapchainImageViews[imageIndex];
    colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentInfo.clearValue = clearColor;

    VkRenderingAttachmentInfo depthAttachmentInfo{};
    depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachmentInfo.imageView = depthImageView;
    depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachmentInfo.clearValue = clearDepth;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    ImGui::Render();
    if (!isLoading) {
        draw(commandBuffer);
    } else {
        ImDrawData* drawData = ImGui::GetDrawData();
        if (drawData) ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
    }

    vkCmdEndRendering(commandBuffer);

    // Transition swapchain image: COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR
    {
        VkImageMemoryBarrier2 presentBarrier{};
        presentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        presentBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        presentBarrier.dstAccessMask = 0;
        presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentBarrier.image = swapchainImages[imageIndex];
        presentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &presentBarrier;

        vkCmdPipelineBarrier2(commandBuffer, &depInfo);
    }

    // End recording commands
    VkResult endCmdResult = vkEndCommandBuffer(commandBuffer);
    if (endCmdResult != VK_SUCCESS) {
        std::cerr << "vkEndCommandBuffer failed: " << endCmdResult << std::endl;
        return;
    }

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    {
        // Serialize pre-apply and submission to ensure tracked layouts match
        // the submission order. Acquire `graphicsSubmitMutex` first.
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);

        // Assign a submit id for this frame submission for diagnostics and
        // expose it via g_cmdSubmitMap so preApply/apply logging can show it.
        uint64_t submitId = g_submitCounter.fetch_add(1);
        {
            std::lock_guard<std::mutex> cmdlk(pendingCmdMutex);
            g_cmdSubmitMap[commandBuffer] = submitId;
        }

        // Log submit details (verbose — disabled for production)
#if 0
        std::cerr << "[VulkanApp] frame submitId=" << submitId
                  << " cmd=" << (void*)commandBuffer
                  << " imageIndex=" << imageIndex
                  << " image=" << (void*)swapchainImages[imageIndex]
                  << " waitCount=" << localWaitSemaphores.size();
        for (size_t _i = 0; _i < localWaitSemaphores.size(); ++_i) {
            std::cerr << " wait[" << _i << "]=" << (void*)localWaitSemaphores[_i]
                      << "/" << localWaitStages[_i];
        }
        std::cerr << std::endl;
#endif

        // Promote pending layout updates for this submission so validation
        // sees populated layouts for affected subresources.
        preApplyPendingLayoutsBeforeSubmit(commandBuffer);

        r = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
        // Register pending layout updates recorded into this frame's command
        // buffer so they are applied when the frame fence signals.
        deferDestroyUntilFence(inFlightFences[currentFrame], [this, commandBuffer]() {
            applyPendingLayoutUpdatesForCommandBuffer(commandBuffer);
        });
    }
    if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS) {
        std::cerr << "vkQueueSubmit failed: " << r << std::endl;
        return;
    }

    // Cleanup any semaphores that were associated with earlier frames and are now safe to destroy
    for (auto it = semaphoresPendingDestroy.begin(); it != semaphoresPendingDestroy.end(); ) {
        VkSemaphore s = it->first;
        VkFence f = it->second;
        VkResult st = vkGetFenceStatus(device, f);
        if (st == VK_SUCCESS) {
            // frame fence signaled -> safe to destroy semaphore (unregister first)
            resources.removeSemaphore(s);
            vkDestroySemaphore(device, s, nullptr);
            it = semaphoresPendingDestroy.erase(it);
        } else {
            ++it;
        }
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    {
        std::lock_guard<std::mutex> lock(graphicsSubmitMutex);
        r = vkQueuePresentKHR(presentQueue, &presentInfo);
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || framebufferResized || vsyncChanged) {
        framebufferResized = false;
        vsyncChanged = false;
        recreateSwapchain();
        return;
    } else if (r == VK_ERROR_DEVICE_LOST) {
        return;
    } else if (r != VK_SUCCESS) {
        std::cerr << "vkQueuePresentKHR failed: " << r << std::endl;
        return;
    }

    // Process pending command buffers and semaphores cleanup now that present is done
    processPendingCommandBuffers();
    for (auto it = semaphoresPendingDestroy.begin(); it != semaphoresPendingDestroy.end();) {
        VkSemaphore s = it->first; VkFence f = it->second;
        VkResult st = vkGetFenceStatus(device, f);
        if (st == VK_SUCCESS) {
            resources.removeSemaphore(s);
            vkDestroySemaphore(device, s, nullptr);
            it = semaphoresPendingDestroy.erase(it);
        } else ++it;
    }
    //std::cerr << "presented image " << imageIndex << "\n";

    // Hook for derived apps to run post-submit instrumentation (e.g., readback)
    postSubmit();

    // Advance to next CPU frame
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanApp::cleanupSwapchain() {
    // Defer destruction to VulkanResourceManager; clear local handles for swapchain image views
    // Explicitly unregister and destroy swapchain image views now to ensure they're
    // released before device teardown (avoid manager ordering surprises).
    for (auto &iv : swapchainImageViews) {
        if (iv != VK_NULL_HANDLE) {
            if (resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr);
            iv = VK_NULL_HANDLE;
        }
    }
    swapchainImageViews.clear();

    // destroy depth resources
    // Unregister and destroy depth resources now
    if (depthImageView != VK_NULL_HANDLE) {
        if (resources.removeImageView(depthImageView)) vkDestroyImageView(device, depthImageView, nullptr);
        depthImageView = VK_NULL_HANDLE;
    }
    destroyImageWithVma(depthImage, depthImageAllocation, depthImageMemory);
    depthImage = VK_NULL_HANDLE;
    depthImageAllocation = VK_NULL_HANDLE;
    depthImageMemory = VK_NULL_HANDLE;

    // free command buffers from their per-frame pools (not the main commandPool)
    if (!commandBuffers.empty()) {
        // Acquire graphicsSubmitMutex first to maintain consistent lock ordering
        std::lock_guard<std::mutex> lockQ(graphicsSubmitMutex);
        std::lock_guard<std::mutex> lockC(commandPoolMutex);
        for (uint32_t i = 0; i < static_cast<uint32_t>(commandBuffers.size()); ++i) {
            if (commandBuffers[i] != VK_NULL_HANDLE && i < frameCommandPools.size()) {
                vkFreeCommandBuffers(device, frameCommandPools[i], 1, &commandBuffers[i]);
            }
        }
        commandBuffers.clear();
    }

    // destroy swapchain
    if (swapchain != VK_NULL_HANDLE) {
        auto fp = (PFN_vkDestroySwapchainKHR)vkGetInstanceProcAddr(instance, "vkDestroySwapchainKHR");
        if (fp) fp(device, swapchain, nullptr);
        else vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    swapchainImages.clear();
    imagesInFlight.clear();
}

void VulkanApp::recreateSwapchain() {
    int width = 0, height = 0;
    // wait for non-zero size (window might be minimized)
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    VkResult waitResult = deviceWaitIdle();
    if (waitResult == VK_ERROR_DEVICE_LOST) {
        return;
    }
    
    // Wait for all in-flight fences before clearing tracking
    for (size_t i = 0; i < inFlightFences.size(); i++) {
        vkWaitForFences(device, 1, &inFlightFences[i], VK_TRUE, UINT64_MAX);
    }
    
    cleanupSwapchain();


    createSwapchain();
    createImageViews();
    createCommandPool();
    createDepthResources();
    commandBuffers = createCommandBuffers();

    // Ensure imagesInFlight matches the new swapchain image count
    imagesInFlight.clear();
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);

    // Recreate sync objects (semaphores) with the correct count for the new swapchain.
    // Destroy old semaphores first since cleanupSwapchain preserved them.
    for (auto &s : imageAvailableSemaphores) {
        if (s != VK_NULL_HANDLE) { resources.removeSemaphore(s); vkDestroySemaphore(device, s, nullptr); s = VK_NULL_HANDLE; }
    }
    for (auto &s : renderFinishedSemaphores) {
        if (s != VK_NULL_HANDLE) { resources.removeSemaphore(s); vkDestroySemaphore(device, s, nullptr); s = VK_NULL_HANDLE; }
    }
    imageAvailableSemaphores.clear();
    renderFinishedSemaphores.clear();
    createSyncObjects();

    // Re-create ImGui Vulkan backend objects so they use the updated swapchain image count.
    // ImGui_ImplVulkan_Init stores internal arrays sized by ImageCount; when the swapchain
    // changes we must reinit the backend.
    // Destroy + recreate the imgui descriptor pool so the new backend starts with a
    // fresh, uncorrupted pool (reusing the old pool across Shutdown/Init cycles has
    // been observed to produce stale descriptor-set handles on some drivers).
    //
    // Order matters:
    //   1. preImGuiShutdown  — frees our AddTexture descriptors (shadow + widgets)
    //   2. ImGui_ImplVulkan_Shutdown — frees the font descriptor internally via
    //      DestroyFontsTexture → RemoveTexture → vkFreeDescriptorSets
    //   3. Destroy old pool   — safe after all descriptors are freed
    //   4. Create new pool    — fresh, uncorrupted pool
    //   5. Init + CreateFontsTexture
    //   6. onImGuiRecreated   — re-allocate shadow cascade descriptors from new pool
    preImGuiShutdown();
    ImGui_ImplVulkan_Shutdown();
    if (imguiDescriptorPool != VK_NULL_HANDLE) {
        std::cerr << "[VulkanApp::recreateSwapchain] Destroying old imguiDescriptorPool=" << (void*)imguiDescriptorPool << std::endl;
        resources.removeDescriptorPool(imguiDescriptorPool);
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
        imguiDescriptorPool = VK_NULL_HANDLE;
    }

    // Create a fresh imgui descriptor pool
    std::cerr << "[VulkanApp::recreateSwapchain] Creating new imgui descriptor pool ..." << std::endl;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * (uint32_t)std::size(pool_sizes);
        pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
        pool_info.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to recreate ImGui descriptor pool during swapchain recreation!");
        }
        resources.addDescriptorPool(imguiDescriptorPool, "VulkanApp: imguiDescriptorPool (recreated)");
    }

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = findQueueFamilies(physicalDevice).graphicsFamily.value();
    init_info.Queue = graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imguiDescriptorPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<uint32_t>(swapchainImages.size());
    init_info.Allocator = nullptr;
    init_info.MinAllocationSize = 1024 * 1024; // Pad to 1MB to suppress validation small-allocation warnings
    init_info.CheckVkResultFn = [](VkResult err) {
        if (err != VK_SUCCESS) {
            std::cerr << "[ImGui] Vulkan error: " << err << std::endl;
            abort();
        }
    };
    VkPipelineRenderingCreateInfo imguiPipelineRenderingInfo{};
    imguiPipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    imguiPipelineRenderingInfo.colorAttachmentCount = 1;
    imguiPipelineRenderingInfo.pColorAttachmentFormats = &swapchainImageFormat;
    imguiPipelineRenderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    imguiPipelineRenderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = imguiPipelineRenderingInfo;
    init_info.UseDynamicRendering = true;


    bool imguiInitOk = ImGui_ImplVulkan_Init(&init_info);
    // printf("[ImGui] ImGui_ImplVulkan_Init (recreate) returned %s\\n", imguiInitOk ? "true" : "false");
    // Fonts are uploaded automatically by the backend on first NewFrame()
    if (!imguiInitOk) {
        printf("[ImGui] ERROR: ImGui_ImplVulkan_Init (recreate) failed!\n");
    }

    // Notify derived app to re-create any ImGui AddTexture DS that used the old DSL.
    // The Shutdown() above destroyed the old DescriptorSetLayout; DS allocated with it
    // must be freed and re-created with the new DSL to pass validation.
    onImGuiRecreated();

    // Notify derived app to recreate size-dependent offscreen resources
    onSwapchainResized(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    framebufferResized = false;
}

void VulkanApp::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->framebufferResized = true;
    }
}

void VulkanApp::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("Validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Starter";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // extensions
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        // Enable synchronization validation to catch sync errors.
        // GPU-assisted validation is disabled because it forces hardware
        // features (bufferDeviceAddress, scalarBlockLayout, etc.) that
        // integrated GPUs like RADV RENOIR don't support.
        VkValidationFeatureEnableEXT enabledValidationFeatures[] = {
            VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
        };
        VkValidationFeaturesEXT validationFeatures{};
        validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(sizeof(enabledValidationFeatures)/sizeof(enabledValidationFeatures[0]));
        validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;
        createInfo.pNext = &validationFeatures;
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

bool VulkanApp::checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool found = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                found = true;
                break;
            }
        }

        if (!found) return false;
    }

    return true;
}

void VulkanApp::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        if (func(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
}

void VulkanApp::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

QueueFamilyIndices VulkanApp::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    // Prefer a dedicated transfer-only queue family if available (has TRANSFER bit but not GRAPHICS)
    for (uint32_t j = 0; j < queueFamilies.size(); ++j) {
        const auto &qf = queueFamilies[j];
        if ((qf.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.transferFamily = j;
            break;
        }
    }
    // If no dedicated transfer-only family found, fall back to any family that supports transfer
    if (!indices.transferFamily.has_value()) {
        for (uint32_t j = 0; j < queueFamilies.size(); ++j) {
            const auto &qf = queueFamilies[j];
            if (qf.queueFlags & VK_QUEUE_TRANSFER_BIT) {
                indices.transferFamily = j;
                break;
            }
        }
    }

    // Prefer a single graphics+present family, but keep scanning so transfer
    // discovery above is never bypassed.
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.graphicsFamily = i;
                indices.presentFamily = i;
                break;
            }
        }
    }

    // If no combined graphics+present family was found, accept separate families.
    for (uint32_t i = 0; i < queueFamilies.size() && !indices.isComplete(); ++i) {
        const auto& queueFamily = queueFamilies[i];
        if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && !indices.graphicsFamily.has_value()) {
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport && !indices.presentFamily.has_value()) {
            indices.presentFamily = i;
        }
    }

    return indices;
}

void VulkanApp::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& dev : devices) {
        if (isDeviceSuitable(dev)) {
            physicalDevice = dev;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

bool VulkanApp::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    return indices.isComplete();
}

void VulkanApp::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    if (indices.transferFamily.has_value()) uniqueQueueFamilies.insert(indices.transferFamily.value());

    // Query available queue counts for families so we don't request more queues than supported
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> familyProps(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, familyProps.data());

    // Request one or more queues from each unique queue family. If the
    // graphics family supports multiple queues, request extra queues for
    // vegetation and geometry work. Ensure we supply a priorities array
    // with the correct length for each VkDeviceQueueCreateInfo (one float
    // per requested queue) to avoid undefined behavior.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    // container to hold per-create-info priority arrays (must outlive createInfo usage)
    std::vector<std::vector<float>> queuePrioritiesStorage;
    float defaultPriority = 1.0f;
    // track how many queues we requested per family
    std::unordered_map<uint32_t, uint32_t> requestedQueueCount;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        // determine desired queue count: try to allocate 3 from graphics family
        uint32_t want = 1;
        if (queueFamily == indices.graphicsFamily.value()) want = 3; // graphics + vegetation + geometry
        uint32_t available = 1;
        if (queueFamily < familyProps.size()) available = familyProps[queueFamily].queueCount;
        uint32_t take = std::min(available, want);
        if (take == 0) {
            throw std::runtime_error("createLogicalDevice: requested queue family has no available queues (no fallback allowed)");
        }
        queueCreateInfo.queueCount = take;
        // prepare priority array for this create info
        queuePrioritiesStorage.emplace_back(queueCreateInfo.queueCount, defaultPriority);
        queueCreateInfo.pQueuePriorities = queuePrioritiesStorage.back().data();
        queueCreateInfos.push_back(queueCreateInfo);
        requestedQueueCount[queueFamily] = queueCreateInfo.queueCount;
    }

    // Query supported features and enable non-solid fill (wireframe) if available
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);

    VkPhysicalDeviceFeatures deviceFeatures{};
    if (supportedFeatures.fillModeNonSolid) {
        deviceFeatures.fillModeNonSolid = VK_TRUE;
    }
    // Enable wide lines if supported so pipelines can use line widths > 1.0
    if (supportedFeatures.wideLines) {
        deviceFeatures.wideLines = VK_TRUE;
    }
    // Enable geometry shader if supported
    if (supportedFeatures.geometryShader) {
        deviceFeatures.geometryShader = VK_TRUE;
    } else {
        throw std::runtime_error("Selected GPU does not support geometry shaders, but they are required.");
    }
    deviceFeatures.tessellationShader = VK_TRUE;
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    // Robust buffer access: out-of-bounds reads return 0 instead of undefined behavior
    deviceFeatures.robustBufferAccess = VK_TRUE;
    // Enable depth clamp so tessellation-displaced vertices beyond the far plane
    // are clamped instead of clipped (prevents ragged edges at far distance)
    if (supportedFeatures.depthClamp) {
        deviceFeatures.depthClamp = VK_TRUE;
    }
    // Enable multi-draw indirect for GPU-driven rendering
    if (supportedFeatures.multiDrawIndirect) {
        deviceFeatures.multiDrawIndirect = VK_TRUE;
        deviceFeatures.drawIndirectFirstInstance = VK_TRUE;
    }

    // Enable Vulkan 1.1 shaderDrawParameters for gl_BaseInstanceARB in shaders
    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.shaderDrawParameters = VK_TRUE;

    // Mesa RADV ignores VkPhysicalDeviceVulkan13Features — use KHR extension structs instead.
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features{};
    sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    sync2Features.pNext = nullptr;
    sync2Features.synchronization2 = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynRenderFeatures{};
    dynRenderFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    dynRenderFeatures.pNext = &sync2Features;
    dynRenderFeatures.dynamicRendering = VK_TRUE;

    vulkan11Features.pNext = &dynRenderFeatures;

    // Enable Vulkan 1.2 features: drawIndirectCount + descriptorIndexing.
    // timelineSemaphore is part of VkPhysicalDeviceVulkan12Features (core 1.2).
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan11Features;
    vulkan12Features.drawIndirectCount = VK_TRUE;
    vulkan12Features.descriptorIndexing = VK_TRUE;
    vulkan12Features.timelineSemaphore = VK_TRUE;
    vulkan12Features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan12Features;  // Chain Vulkan 1.2 + 1.1 + dynamic rendering + sync2
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;


    // No extension dependency logic needed
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    // Device layers are deprecated since Vulkan 1.0 — must be 0.
    // Validation layers are enabled at instance creation only.
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;


    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    // Create main descriptor pool immediately after device creation
    // (choose reasonable default counts for UBOs and samplers)
    createDescriptorPool(32, 32);

    // retrieve queue handles. If we requested multiple queues from the
    // graphics family, obtain them; otherwise fall back to the main
    // graphics queue for vegetation/geometry work. Also obtain transferQueue if available.
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    uint32_t gfxRequested = 1;
    auto it = requestedQueueCount.find(indices.graphicsFamily.value());
    if (it != requestedQueueCount.end()) gfxRequested = it->second;
    if (gfxRequested > 1) vkGetDeviceQueue(device, indices.graphicsFamily.value(), 1, &vegetationQueue); else vegetationQueue = graphicsQueue;
    if (gfxRequested > 2) vkGetDeviceQueue(device, indices.graphicsFamily.value(), 2, &geometryQueue); else geometryQueue = graphicsQueue;
    if (indices.presentFamily.value() == indices.graphicsFamily.value()) {
        // present uses the same family; reuse the main graphics queue
        presentQueue = graphicsQueue;
    } else {
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    if (indices.transferFamily.has_value()) {
        if (indices.transferFamily.value() == indices.graphicsFamily.value()) {
            transferQueue = graphicsQueue;
        } else {
            vkGetDeviceQueue(device, indices.transferFamily.value(), 0, &transferQueue);
        }
    } else {
        transferQueue = graphicsQueue;
    }
    // debug: print queue family indices and whether the queue handles are shared
    std::cerr << "createLogicalDevice: graphicsFamily=" << indices.graphicsFamily.value()
              << " presentFamily=" << indices.presentFamily.value()
              << " transferFamily="
              << (indices.transferFamily.has_value() ? std::to_string(indices.transferFamily.value()) : std::string("none"))
              << "\n";
    std::cerr << "graphicsQueue handle: " << graphicsQueue
              << " presentQueue handle: " << presentQueue
              << " transferQueue handle: " << transferQueue
              << " dedicatedTransfer=" << (transferQueue != graphicsQueue ? "yes" : "no")
              << "\n";

    // Initialize the persistent staging ring buffer for async uploads
    stagingRing.init(device, physicalDevice);
}

int VulkanApp::getWidth() {
    return swapchainExtent.width;
}

int VulkanApp::getHeight() {
    return swapchainExtent.height;
}

GLFWwindow* VulkanApp::getWindow() {
    return window;
}


void VulkanApp::run() {
    initWindow();
    initVulkan();

    // Fonts are now loaded — render one frame showing the loading screen before
    // the (potentially slow) setup() call so the user sees something immediately.
    isLoading = true;
    glfwPollEvents();
    drawFrame();

    setup();
    isLoading = false;

    mainLoop();  
    cleanup();
}
