#include "DescriptorSetBuilder.hpp"
#include <cstring>
DescriptorSetBuilder::DescriptorSetBuilder(VulkanApp* app_, ShadowMapper* shadowMapper_)
    : app(app_), shadow(shadowMapper_) {}

VkDescriptorSet DescriptorSetBuilder::createMainDescriptorSet(const Triple& tr, Buffer& uniformBuffer, bool bindMaterial, Buffer* materialBuffer, VkDeviceSize materialOffset, Buffer* instanceBuffer) {
    VkDescriptorSet ds = app->createDescriptorSet(app->getDescriptorSetLayout());

    VkDescriptorBufferInfo bufferInfo{ uniformBuffer.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet uboWrite{}; uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; uboWrite.dstSet = ds; uboWrite.dstBinding = 0; uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; uboWrite.descriptorCount = 1; uboWrite.pBufferInfo = &bufferInfo;

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

void DescriptorSetBuilder::updateMaterialBinding(std::vector<VkDescriptorSet>& sets, Buffer& materialBuffer, VkDeviceSize elementSize) {
    // Write material storage buffer into each provided descriptor set at binding 5.
    for (size_t i = 0; i < sets.size(); ++i) {
        VkDescriptorBufferInfo materialBufInfo{ materialBuffer.buffer, static_cast<VkDeviceSize>(i) * elementSize, elementSize };
        VkWriteDescriptorSet materialWrite{};
        materialWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        materialWrite.dstSet = sets[i];
        materialWrite.dstBinding = 5;
        materialWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        materialWrite.descriptorCount = 1;
        materialWrite.pBufferInfo = &materialBufInfo;
        app->updateDescriptorSet(sets[i], { materialWrite });
    }
}

// Note: functions accept an `instanceBuffer` parameter for backward compatibility
// but do not currently use it. Callers may pass nullptr.
