#pragma once

#include "VulkanApp.hpp"
#include "DescriptorSetBuilder.hpp"
#include "TextureArrayManager.hpp"
#include "TextureTriple.hpp"
#include "MaterialManager.hpp"
#include "ShaderStage.hpp"
#include "../utils/FileReader.hpp"
#include "../math/Vertex.hpp"
#include <unordered_map>
#include <memory>
#include "Model3DVersion.hpp"
#include "SkyRenderer.hpp"
#include "SkySphere.hpp"
#include "VertexBufferObject.hpp"
#include "VertexBufferObjectBuilder.hpp"
#include "IndirectRenderer.hpp"
#include "ShadowMapper.hpp"
#include "WaterRenderer.hpp"
#include "../math/SphereModel.hpp"
#include "../Uniforms.hpp"
#include "../utils/Scene.hpp"
// forward declare SkyWidget to avoid including widget headers here
class SkyWidget;

class SceneRenderer {
public:
    VulkanApp* app = nullptr;

    std::unordered_map<NodeID, Model3DVersion> nodeModelVersions;
    std::unordered_map<NodeID, Model3DVersion> waterNodeModelVersions;

    std::unique_ptr<SkyRenderer> skyRenderer;
    std::unique_ptr<SkySphere> skySphere;

    VertexBufferObject skyVBO;

    IndirectRenderer indirectRenderer;
    ShadowMapper shadowMapper;
    WaterRenderer waterRenderer;

    // Graphics pipelines owned by SceneRenderer
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline waterPipeline = VK_NULL_HANDLE;  // Water pipeline with depth test but no depth write
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    VkPipeline depthPrePassPipeline = VK_NULL_HANDLE;

    // Create graphics pipelines. This mirrors the previous logic in main.cpp
    void createPipelines() {
        if (!app) return;

        ShaderStage vertexShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.vert.spv")),
            VK_SHADER_STAGE_VERTEX_BIT
        );

        ShaderStage fragmentShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.frag.spv")),
            VK_SHADER_STAGE_FRAGMENT_BIT
        );

        ShaderStage tescShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.tesc.spv")),
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
        );
        ShaderStage teseShader = ShaderStage(
            app->createShaderModule(FileReader::readFile("shaders/main.tese.spv")),
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
        );

        graphicsPipeline = app->createGraphicsPipeline(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShader.info
            },
            VkVertexInputBindingDescription {
                0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX
            },
            {
                VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
            },
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, false, true, VK_COMPARE_OP_EQUAL
        );

        // Expose the main graphics pipeline to app helpers
        app->setAppGraphicsPipeline(graphicsPipeline);

        graphicsPipelineWire = app->createGraphicsPipeline(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShader.info
            },
            VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
            {
                VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
            },
            VK_POLYGON_MODE_LINE,
            VK_CULL_MODE_BACK_BIT,
            false,
            true,
            VK_COMPARE_OP_EQUAL
        );

        depthPrePassPipeline = app->createGraphicsPipeline(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShader.info
            },
            VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
            {
                VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
            },
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_BACK_BIT,
            true, // depthWrite
            false // colorWrite disabled
        );

        waterPipeline = app->createGraphicsPipeline(
            {
                vertexShader.info,
                tescShader.info,
                teseShader.info,
                fragmentShader.info
            },
            VkVertexInputBindingDescription { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX },
            {
                VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
                VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                VkVertexInputAttributeDescription { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord) },
                VkVertexInputAttributeDescription { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                VkVertexInputAttributeDescription { 5, 0, VK_FORMAT_R32_SINT, offsetof(Vertex, texIndex) }
            },
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            false, // depthWrite disabled
            true,
            VK_COMPARE_OP_LESS_OR_EQUAL
        );

        // Destroy shader modules
        vkDestroyShaderModule(app->getDevice(), teseShader.info.module, nullptr);
        vkDestroyShaderModule(app->getDevice(), tescShader.info.module, nullptr);
        vkDestroyShaderModule(app->getDevice(), fragmentShader.info.module, nullptr);
        vkDestroyShaderModule(app->getDevice(), vertexShader.info.module, nullptr);
    }

    // Create descriptor sets (moved from main.cpp). Requires caller to have populated materials
    // and ensured tripleCount reflects the number of texture triples.
    void createDescriptorSets(class MaterialManager &materialManager, class TextureArrayManager &textureArrayManager, VkDescriptorSet &descriptorSetOut, VkDescriptorSet &shadowPassDescriptorSetOut, size_t tripleCount) {
        if (!app) return;

        if (tripleCount == 0) tripleCount = 1;

        // Material buffer is expected to be created by caller (updateMaterials)

        uint32_t estimatedPerTextureSets = static_cast<uint32_t>(tripleCount * 4);
        uint32_t estimatedModelObjects = static_cast<uint32_t>(1 + 2 * tripleCount);
        uint32_t estimatedPerInstanceSets = estimatedModelObjects * 2;
        uint32_t totalSets = estimatedPerTextureSets + estimatedPerInstanceSets + 4 + 1;

        app->createDescriptorPool(totalSets, totalSets * 4);

        // Create global material descriptor set
        VkDescriptorSet globalMatDS = VK_NULL_HANDLE;
        if (materialManager.count() > 0) {
            globalMatDS = app->createDescriptorSet(app->getMaterialDescriptorSetLayout());
            VkDescriptorBufferInfo materialBufInfo{ materialManager.getBuffer().buffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet matWrite{}; matWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; matWrite.dstSet = globalMatDS; matWrite.dstBinding = 5; matWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; matWrite.descriptorCount = 1; matWrite.pBufferInfo = &materialBufInfo;
            app->updateDescriptorSet(globalMatDS, { matWrite });
            app->setMaterialDescriptorSet(globalMatDS);
            app->registerDescriptorSet(globalMatDS);
        }

        descriptorSetOut = VK_NULL_HANDLE;
        DescriptorSetBuilder dsBuilder(app, &shadowMapper);
        for (size_t i = 0; i < tripleCount; ++i) {
            Triple tr;
            tr.albedo.view = textureArrayManager.albedoArray.view;
            tr.albedoSampler = textureArrayManager.albedoSampler;
            tr.normal.view = textureArrayManager.normalArray.view;
            tr.normalSampler = textureArrayManager.normalSampler;
            tr.height.view = textureArrayManager.bumpArray.view;
            tr.heightSampler = textureArrayManager.bumpSampler;

            if (descriptorSetOut == VK_NULL_HANDLE) {
                VkDescriptorSet ds = dsBuilder.createMainDescriptorSet(tr, mainPassUBO.buffer, false, nullptr, 0);

                VkDescriptorBufferInfo skyBufInfo{ mainPassUBO.buffer.buffer, 0, sizeof(UniformObject) };
                VkWriteDescriptorSet skyWrite{}; skyWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; skyWrite.dstSet = ds; skyWrite.dstBinding = 6; skyWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; skyWrite.descriptorCount = 1; skyWrite.pBufferInfo = &skyBufInfo;

                VkDescriptorBufferInfo waterParamsBufInfo{ waterPassUBO.buffer.buffer, 0, sizeof(WaterParamsGPU) };
                VkWriteDescriptorSet waterParamsWrite{}; waterParamsWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; waterParamsWrite.dstSet = ds; waterParamsWrite.dstBinding = 7; waterParamsWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; waterParamsWrite.descriptorCount = 1; waterParamsWrite.pBufferInfo = &waterParamsBufInfo;

                app->updateDescriptorSet(ds, { skyWrite, waterParamsWrite });
                descriptorSetOut = ds;
                app->registerDescriptorSet(descriptorSetOut);
            }

            if (shadowPassDescriptorSetOut == VK_NULL_HANDLE) {
                VkDescriptorSet ds = dsBuilder.createShadowDescriptorSet(tr, shadowPassUBO.buffer, false, nullptr, 0);
                shadowPassDescriptorSetOut = ds;
                app->registerDescriptorSet(shadowPassDescriptorSetOut);
            }
        }

        if (textureArrayManager.layerAmount > 0) {
            VkDescriptorImageInfo albedoArrayInfo{ textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo normalArrayInfo{ textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo bumpArrayInfo{ textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            if (descriptorSetOut != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = descriptorSetOut; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = descriptorSetOut; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = descriptorSetOut; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                app->updateDescriptorSet(descriptorSetOut, { w1, w2, w3 });
            }
            if (shadowPassDescriptorSetOut != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1.dstSet = shadowPassDescriptorSetOut; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w1.descriptorCount = 1; w1.pImageInfo = &albedoArrayInfo;
                VkWriteDescriptorSet w2{}; w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet = shadowPassDescriptorSetOut; w2.dstBinding = 2; w2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.descriptorCount = 1; w2.pImageInfo = &normalArrayInfo;
                VkWriteDescriptorSet w3{}; w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w3.dstSet = shadowPassDescriptorSetOut; w3.dstBinding = 3; w3.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w3.descriptorCount = 1; w3.pImageInfo = &bumpArrayInfo;
                app->updateDescriptorSet(shadowPassDescriptorSetOut, { w1, w2, w3 });
            }
        }
    }

    template<typename T>
    struct PassUBO {
        Buffer buffer;
        T data;

        PassUBO() = default;
        PassUBO(VulkanApp* app, size_t size) {
            buffer = app->createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
    };

    PassUBO<UniformObject> mainPassUBO{};
    PassUBO<UniformObject> shadowPassUBO{};
    PassUBO<WaterParamsGPU> waterPassUBO{};

    SceneRenderer(VulkanApp* app_) : app(app_), shadowMapper(app_, 8192), waterRenderer(app_) {}

    void init(VulkanApp* app_, SkyWidget* skyWidget = nullptr, VkDescriptorSet descriptorSet = VK_NULL_HANDLE) {
        if (app_) app = app_;
        if (!app) return;

        // Sky renderer
        if (!skyRenderer) {
            skyRenderer = std::make_unique<SkyRenderer>(app);
            skyRenderer->init();
        }

        // Build a sphere VBO for sky rendering if not created
        if (skyVBO.vertexBuffer.buffer == VK_NULL_HANDLE && skyVBO.indexCount == 0) {
            SphereModel sphere(0.5f, 32, 16, 0);
            skyVBO = VertexBufferObjectBuilder::create(app, sphere);
        }

        // Shadow mapper
        // Note: init() on shadowMapper is safe to call if already initialized
        shadowMapper.init();

        // Indirect renderer (compute culling + merged buffers)
        indirectRenderer.init(app);

        // Create global UBOs if not already created
        if (mainPassUBO.buffer.buffer == VK_NULL_HANDLE) mainPassUBO = PassUBO<UniformObject>(app, sizeof(UniformObject));
        if (shadowPassUBO.buffer.buffer == VK_NULL_HANDLE) shadowPassUBO = PassUBO<UniformObject>(app, sizeof(UniformObject));
        if (waterPassUBO.buffer.buffer == VK_NULL_HANDLE) waterPassUBO = PassUBO<WaterParamsGPU>(app, sizeof(WaterParamsGPU));

        // Initialize water renderer with its UBO buffer
        waterRenderer.init(waterPassUBO.buffer);

        // If skyWidget and descriptorSet are provided, initialize SkySphere now
        if (skyWidget && descriptorSet != VK_NULL_HANDLE && !skySphere) {
            skySphere = std::make_unique<SkySphere>(app);
            skySphere->init(skyWidget, descriptorSet);
        }
    }

    void cleanup() {
        if (!app) return;

        if (skyRenderer) skyRenderer->cleanup();
        skyVBO.destroy(app->getDevice());

        // Remove meshes registered in indirect renderer
        for (auto &entry : nodeModelVersions) {
            if (entry.second.meshId != UINT32_MAX) indirectRenderer.removeMesh(entry.second.meshId);
        }
        nodeModelVersions.clear();

        // Cleanup indirect renderer resources
        indirectRenderer.cleanup(app);

        // Remove water meshes registered in WaterRenderer
        for (auto &entry : waterNodeModelVersions) {
            if (entry.second.meshId != UINT32_MAX) waterRenderer.getIndirectRenderer().removeMesh(entry.second.meshId);
        }
        waterNodeModelVersions.clear();

        // Cleanup water renderer
        waterRenderer.cleanup();

        // Destroy UBO buffers
        if (mainPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), mainPassUBO.buffer.buffer, nullptr);
        if (mainPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), mainPassUBO.buffer.memory, nullptr);
        if (shadowPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), shadowPassUBO.buffer.buffer, nullptr);
        if (shadowPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), shadowPassUBO.buffer.memory, nullptr);
        if (waterPassUBO.buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(app->getDevice(), waterPassUBO.buffer.buffer, nullptr);
        if (waterPassUBO.buffer.memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), waterPassUBO.buffer.memory, nullptr);

        // Shadow mapper and sky sphere cleanup
        shadowMapper.cleanup();
        if (skySphere) skySphere->cleanup();
    }

    // initSky removed; use init(app, skyWidget, descriptorSet) instead
};
