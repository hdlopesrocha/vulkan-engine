#include "DescriptorSetBuilder.hpp"
#include "../Uniforms.hpp"
#include <cstring>
#include <iostream>
DescriptorSetBuilder::DescriptorSetBuilder(VulkanApp* app_, ShadowRenderer* shadowMapper_)
    : app(app_), shadow(shadowMapper_) {}

VkDescriptorSet DescriptorSetBuilder::createMainDescriptorSet(const Triple& tr, Buffer& uniformBuffer, bool bindMaterial, Buffer* materialBuffer, VkDeviceSize materialOffset, Buffer* instanceBuffer) {
    VkDescriptorSet ds = app->createDescriptorSet(app->getDescriptorSetLayout());

    std::cout << "[DEBUG] DescriptorSetBuilder: Creating descriptor set with UBO buffer=" 
              << uniformBuffer.buffer << " memory=" << uniformBuffer.memory << std::endl;

    VkDescriptorBufferInfo bufferInfo{ uniformBuffer.buffer, 0, sizeof(UniformObject) };
    VkWriteDescriptorSet uboWrite{}; uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; uboWrite.dstSet = ds; uboWrite.dstBinding = 0; uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; uboWrite.descriptorCount = 1; uboWrite.pBufferInfo = &bufferInfo;

    std::cout << "[DEBUG] UBO Write: dstSet=" << uboWrite.dstSet << " dstBinding=" << uboWrite.dstBinding 
              << " buffer=" << bufferInfo.buffer << " offset=" << bufferInfo.offset 
              << " range=" << bufferInfo.range << std::endl;

    VkDescriptorImageInfo imageInfo{ tr.albedoSampler, tr.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet samplerWrite{}; samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite.dstSet = ds; samplerWrite.dstBinding = 1; samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite.descriptorCount = 1; samplerWrite.pImageInfo = &imageInfo;

    VkDescriptorImageInfo normalInfo{ tr.normalSampler, tr.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet normalWrite{}; normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; normalWrite.dstSet = ds; normalWrite.dstBinding = 2; normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; normalWrite.descriptorCount = 1; normalWrite.pImageInfo = &normalInfo;

    VkDescriptorImageInfo heightInfo{ tr.heightSampler, tr.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet heightWrite{}; heightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; heightWrite.dstSet = ds; heightWrite.dstBinding = 3; heightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; heightWrite.descriptorCount = 1; heightWrite.pImageInfo = &heightInfo;

    VkDescriptorImageInfo shadowInfo{ shadow->getShadowMapSampler(), shadow->getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet shadowWrite{}; shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; shadowWrite.dstSet = ds; shadowWrite.dstBinding = 4; shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; shadowWrite.descriptorCount = 1; shadowWrite.pImageInfo = &shadowInfo;

    if (bindMaterial && materialBuffer) {
        VkDescriptorBufferInfo materialBufInfo{ materialBuffer->buffer, materialOffset, VK_WHOLE_SIZE };
        VkWriteDescriptorSet materialWrite{}; materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; materialWrite.dstSet = ds; materialWrite.dstBinding = 5; materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; materialWrite.descriptorCount = 1; materialWrite.pBufferInfo = &materialBufInfo;
        app->updateDescriptorSet(ds, { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite, materialWrite });
    } else {
        app->updateDescriptorSet(ds, { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite });
    }
    return ds;
}

VkDescriptorSet DescriptorSetBuilder::createShadowDescriptorSet(const Triple& tr, Buffer& shadowUniformBuffer, bool bindMaterial, Buffer* materialBuffer, VkDeviceSize materialOffset, Buffer* instanceBuffer) {
    VkDescriptorSet ds = app->createDescriptorSet(app->getDescriptorSetLayout());
    VkDescriptorBufferInfo bufferInfo{ shadowUniformBuffer.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet uboWrite{}; uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; uboWrite.dstSet = ds; uboWrite.dstBinding = 0; uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; uboWrite.descriptorCount = 1; uboWrite.pBufferInfo = &bufferInfo;
    VkDescriptorImageInfo imageInfo{ tr.albedoSampler, tr.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet samplerWrite{}; samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite.dstSet = ds; samplerWrite.dstBinding = 1; samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite.descriptorCount = 1; samplerWrite.pImageInfo = &imageInfo;

    VkDescriptorImageInfo normalInfo{ tr.normalSampler, tr.normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet normalWrite{}; normalWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; normalWrite.dstSet = ds; normalWrite.dstBinding = 2; normalWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; normalWrite.descriptorCount = 1; normalWrite.pImageInfo = &normalInfo;

    VkDescriptorImageInfo heightInfo{ tr.heightSampler, tr.height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet heightWrite{}; heightWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; heightWrite.dstSet = ds; heightWrite.dstBinding = 3; heightWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; heightWrite.descriptorCount = 1; heightWrite.pImageInfo = &heightInfo;

    // Shadow pass: use read-only depth layout for descriptors so sampling operations expect DEPTH_STENCIL_READ_ONLY_OPTIMAL.
    // The shader doesn't actually sample binding 4 during shadow pass (early-out path).
    VkDescriptorImageInfo shadowInfo{ shadow->getShadowMapSampler(), shadow->getShadowMapView(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet shadowWrite{}; shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; shadowWrite.dstSet = ds; shadowWrite.dstBinding = 4; shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; shadowWrite.descriptorCount = 1; shadowWrite.pImageInfo = &shadowInfo;

    if (bindMaterial && materialBuffer) {
        VkDescriptorBufferInfo materialBufInfo{ materialBuffer->buffer, materialOffset, VK_WHOLE_SIZE };
        VkWriteDescriptorSet materialWrite{}; materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; materialWrite.dstSet = ds; materialWrite.dstBinding = 5; materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; materialWrite.descriptorCount = 1; materialWrite.pBufferInfo = &materialBufInfo;
        app->updateDescriptorSet(ds, { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite, materialWrite });
    } else {
        app->updateDescriptorSet(ds, { uboWrite, samplerWrite, normalWrite, heightWrite, shadowWrite });
    }
    return ds;
}

VkDescriptorSet DescriptorSetBuilder::createSphereDescriptorSet(const Triple& tr, Buffer& sphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer, VkDeviceSize materialOffset, Buffer* instanceBuffer) {
    (void)instanceBuffer;
    // Same as main descriptor set but uses sphereUniformBuffer
    return createMainDescriptorSet(tr, sphereUniformBuffer, bindMaterial, materialBuffer, materialOffset, nullptr);
}

VkDescriptorSet DescriptorSetBuilder::createShadowSphereDescriptorSet(const Triple& tr, Buffer& shadowSphereUniformBuffer, bool bindMaterial, Buffer* materialBuffer, VkDeviceSize materialOffset, Buffer* instanceBuffer) {
    (void)instanceBuffer;
    return createShadowDescriptorSet(tr, shadowSphereUniformBuffer, bindMaterial, materialBuffer, materialOffset, nullptr);
}

// Material binding is now managed via a single global descriptor set; per-set updates are removed.

// Note: functions accept an `instanceBuffer` parameter for backward compatibility
// but do not currently use it. Callers may pass nullptr.
