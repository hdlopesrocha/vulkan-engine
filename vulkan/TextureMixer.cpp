#include "TextureMixer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include "PerlinPushConstants.hpp"
#include "TextureArrayManager.hpp"
#include <algorithm>
#include <stdexcept>

uint32_t TextureMixer::getArrayLayerCount() const {
	return textureArrayManager ? textureArrayManager->layerAmount : 0;
}

uint32_t TextureMixer::getLayerWidth() const {
	return textureArrayManager ? textureArrayManager->width : width;
}

uint32_t TextureMixer::getLayerHeight() const {
	return textureArrayManager ? textureArrayManager->height : height;
}

int TextureMixer::getBytesPerPixel() const {
	return 4; // RGBA8
}

TextureMixer::TextureMixer() {}

void TextureMixer::init(VulkanApp* app, uint32_t width, uint32_t height, TextureArrayManager* textureArrayManager) {
	this->app = app;
	this->textureArrayManager = textureArrayManager;
	this->width = width;
	this->height = height;
	// No editable textures created anymore; we will write directly into texture arrays when available

	// Create compute pipeline and descriptor sets so we can generate textures on demand
	createComputePipeline();
	printf("[EditableTextureSet] Compute pipeline created for editable textures\n");
}

// Backwards-compatible overload: callers that don't pass a TextureArrayManager
void TextureMixer::init(VulkanApp* app, uint32_t width, uint32_t height) {
	init(app, width, height, nullptr);
}

// setTextureManager removed — EditableTextureSet creates its own compute sampler
// and compute pipeline during init

void TextureMixer::setOnTextureGenerated(std::function<void()> callback) {
	onTextureGeneratedCallback = callback;
}

void TextureMixer::generateInitialTextures(std::vector<MixerParameters> &mixerParams) {
	printf("Generating initial textures (Albedo, Normal, Bump)...\n");
	for (auto &param : mixerParams) {
		generatePerlinNoise(param);
	}
}


void TextureMixer::cleanup() {
	// No editable textures to cleanup when using global texture arrays

	if (computePipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(app->getDevice(), computePipeline, nullptr);
		computePipeline = VK_NULL_HANDLE;
	}
	if (computePipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(app->getDevice(), computePipelineLayout, nullptr);
		computePipelineLayout = VK_NULL_HANDLE;
	}
	if (computeDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(app->getDevice(), computeDescriptorSetLayout, nullptr);
		computeDescriptorSetLayout = VK_NULL_HANDLE;
	}
	if (computeDescriptorPool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(app->getDevice(), computeDescriptorPool, nullptr);
		computeDescriptorPool = VK_NULL_HANDLE;
	}

	if (computeSampler != VK_NULL_HANDLE) {
		vkDestroySampler(app->getDevice(), computeSampler, nullptr);
		computeSampler = VK_NULL_HANDLE;
	}
}


void TextureMixer::createComputePipeline() {
	// Descriptor layout: three storage images (albedo, normal, bump) and three sampler arrays
	VkDescriptorSetLayoutBinding bindings[6] = {};

	// binding 0: albedo storage image (writeonly)
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// binding 1: albedo sampler2DArray
	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// binding 2: normal sampler2DArray
	bindings[2].binding = 2;
	bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[2].descriptorCount = 1;
	bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// binding 3: bump sampler2DArray
	bindings[3].binding = 3;
	bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[3].descriptorCount = 1;
	bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// binding 4: normal storage image (writeonly)
	bindings[4].binding = 4;
	bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings[4].descriptorCount = 1;
	bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// binding 5: bump storage image (writeonly)
	bindings[5].binding = 5;
	bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings[5].descriptorCount = 1;
	bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 6;
	layoutInfo.pBindings = bindings;

	if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor set layout!");
	}

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(PerlinPushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline layout!");
	}

	auto computeShaderCode = FileReader::readFile("shaders/perlin_noise.comp.spv");
	VkShaderModule computeShaderModule = app->createShaderModule(computeShaderCode);

	VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = computeShaderModule;
	computeShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = computeShaderStageInfo;
	pipelineInfo.layout = computePipelineLayout;

	if (vkCreateComputePipelines(app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline!");
	}

	vkDestroyShaderModule(app->getDevice(), computeShaderModule, nullptr);

	VkDescriptorPoolSize poolSizes[2] = {};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	// triple set uses 3 storage images, plus one per-map storage image each (3)
	poolSizes[0].descriptorCount = 10; // provide extra slack
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	// triple set uses 3 combined samplers, per-map sets use 2 each (6) => total 9
	poolSizes[1].descriptorCount = 20; // provide extra slack for samplers

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	// Provide generous capacity to avoid allocation failures
	poolInfo.maxSets = 10;

	if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor pool!");
	}

	// Create a single descriptor set that binds albedo/normal/bump storage images and samplers
	createTripleComputeDescriptorSet();
	// Also create per-map descriptor sets so single-map generation is possible
	createComputeDescriptorSet(0, albedoComputeDescSet);
	createComputeDescriptorSet(1, normalComputeDescSet);
	createComputeDescriptorSet(2, bumpComputeDescSet);
}

void TextureMixer::createTripleComputeDescriptorSet() {
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = computeDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &computeDescriptorSetLayout;

	if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &tripleComputeDescSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate compute triple descriptor set!");
	}

	// Storage image infos: albedo (binding 0), normal (binding 4), bump (binding 5)
	VkDescriptorImageInfo albedoImageInfo{};
	if (textureArrayManager) albedoImageInfo.imageView = textureArrayManager->albedoArray.view;
	else albedoImageInfo.imageView = VK_NULL_HANDLE;
	albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo normalImageInfo{};
	if (textureArrayManager) normalImageInfo.imageView = textureArrayManager->normalArray.view;
	else normalImageInfo.imageView = VK_NULL_HANDLE;
	normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo bumpImageInfo{};
	if (textureArrayManager) bumpImageInfo.imageView = textureArrayManager->bumpArray.view;
	else bumpImageInfo.imageView = VK_NULL_HANDLE;
	bumpImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	// Sampler infos for the three arrays (binding 1,2,3)
	VkDescriptorImageInfo albedoSamplerInfo{};
	VkDescriptorImageInfo normalSamplerInfo{};
	VkDescriptorImageInfo bumpSamplerInfo{};

	if (textureArrayManager) {
		albedoSamplerInfo.imageView = textureArrayManager->albedoArray.view;
		albedoSamplerInfo.sampler = textureArrayManager->albedoSampler;
		albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		normalSamplerInfo.imageView = textureArrayManager->normalArray.view;
		normalSamplerInfo.sampler = textureArrayManager->normalSampler;
		normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		bumpSamplerInfo.imageView = textureArrayManager->bumpArray.view;
		bumpSamplerInfo.sampler = textureArrayManager->bumpSampler;
		bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	} else {
		// No TextureArrayManager: leave sampler imageViews null (not supported)
		albedoSamplerInfo.imageView = VK_NULL_HANDLE; albedoSamplerInfo.sampler = VK_NULL_HANDLE; albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		normalSamplerInfo.imageView = VK_NULL_HANDLE; normalSamplerInfo.sampler = VK_NULL_HANDLE; normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		bumpSamplerInfo.imageView = VK_NULL_HANDLE; bumpSamplerInfo.sampler = VK_NULL_HANDLE; bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	VkWriteDescriptorSet writes[6] = {};
	// binding 0 - albedo storage image
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = tripleComputeDescSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[0].descriptorCount = 1;
	writes[0].pImageInfo = &albedoImageInfo;

	// binding 1 - albedo sampler array
	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = tripleComputeDescSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[1].descriptorCount = 1;
	writes[1].pImageInfo = &albedoSamplerInfo;

	// binding 2 - normal sampler array
	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = tripleComputeDescSet;
	writes[2].dstBinding = 2;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[2].descriptorCount = 1;
	writes[2].pImageInfo = &normalSamplerInfo;

	// binding 3 - bump sampler array
	writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet = tripleComputeDescSet;
	writes[3].dstBinding = 3;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writes[3].descriptorCount = 1;
	writes[3].pImageInfo = &bumpSamplerInfo;

	// binding 4 - normal storage image
	writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[4].dstSet = tripleComputeDescSet;
	writes[4].dstBinding = 4;
	writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[4].descriptorCount = 1;
	writes[4].pImageInfo = &normalImageInfo;

	// binding 5 - bump storage image
	writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[5].dstSet = tripleComputeDescSet;
	writes[5].dstBinding = 5;
	writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[5].descriptorCount = 1;
	writes[5].pImageInfo = &bumpImageInfo;

	vkUpdateDescriptorSets(app->getDevice(), 6, writes, 0, nullptr);
}

VkDescriptorSet TextureMixer::getPreviewDescriptor(int map) {
	if (textureArrayManager && editableLayer != UINT32_MAX) {
		ImTextureID id = textureArrayManager->getImTexture(editableLayer, map);
		return (VkDescriptorSet)id;
	}
	// Fallback to editable texture descriptor
	return VK_NULL_HANDLE;
}

VkDescriptorSet TextureMixer::getPreviewDescriptor(int map, uint32_t layer) {
	if (textureArrayManager && layer < textureArrayManager->layerAmount) {
		ImTextureID id = textureArrayManager->getImTexture(layer, map);
		return (VkDescriptorSet)id;
	}
	// Fallback to default behavior
	return getPreviewDescriptor(map);
}

void TextureMixer::createComputeDescriptorSet(int map, VkDescriptorSet& descSet) {
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = computeDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &computeDescriptorSetLayout;

	if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &descSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate compute descriptor set!");
	}

	VkDescriptorImageInfo imageInfo{};
	// Storage image for per-map generation: bind the array view (entire array) and use push constant to select layer
	if (textureArrayManager) {
		if (map == 0) imageInfo.imageView = textureArrayManager->albedoArray.view;
		else if (map == 1) imageInfo.imageView = textureArrayManager->normalArray.view;
		else imageInfo.imageView = textureArrayManager->bumpArray.view;
	} else {
		imageInfo.imageView = VK_NULL_HANDLE; // no fallback; caller should provide TextureArrayManager
	}
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pImageInfo = &imageInfo;

	// Ensure we have a sampler to bind as primary/secondary inputs for the compute shader.
	if (computeSampler == VK_NULL_HANDLE) {
		computeSampler = app->createTextureSampler(1);
	}

	VkDescriptorImageInfo samplerInfo{};
	// If a TextureArrayManager was provided, bind its array view and sampler so compute samples from arrays
	if (textureArrayManager) {
		if (map == 0) {
			samplerInfo.imageView = textureArrayManager->albedoArray.view;
			samplerInfo.sampler = textureArrayManager->albedoSampler;
		} else if (map == 1) {
			samplerInfo.imageView = textureArrayManager->normalArray.view;
			samplerInfo.sampler = textureArrayManager->normalSampler;
		} else {
			samplerInfo.imageView = textureArrayManager->bumpArray.view;
			samplerInfo.sampler = textureArrayManager->bumpSampler;
		}
		samplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	} else {
		samplerInfo.imageView = VK_NULL_HANDLE;
		samplerInfo.sampler = VK_NULL_HANDLE;
		samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	VkWriteDescriptorSet samplerWrite1{}; samplerWrite1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite1.dstSet = descSet; samplerWrite1.dstBinding = 1; samplerWrite1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite1.descriptorCount = 1; samplerWrite1.pImageInfo = &samplerInfo;
	VkWriteDescriptorSet samplerWrite2{}; samplerWrite2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite2.dstSet = descSet; samplerWrite2.dstBinding = 2; samplerWrite2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite2.descriptorCount = 1; samplerWrite2.pImageInfo = &samplerInfo;

	VkWriteDescriptorSet writes[3] = { descriptorWrite, samplerWrite1, samplerWrite2 };
	vkUpdateDescriptorSets(app->getDevice(), 3, writes, 0, nullptr);
}

void TextureMixer::generatePerlinNoise(MixerParameters &params) {
	// generate regardless of external texture lists
	if(params.targetLayer == params.primaryTextureIdx || 
	   params.targetLayer == params.secondaryTextureIdx) {
		return; // avoid sampling and writing to the same layer
	}

	// We now perform a single dispatch that writes all three images (albedo, normal, bump)
	VkDescriptorSet descSet = tripleComputeDescSet;
	if (descSet == VK_NULL_HANDLE) return;

	PerlinPushConstants pushConstants;
	pushConstants.scale = params.perlinScale;
	pushConstants.octaves = params.perlinOctaves;
	pushConstants.persistence = params.perlinPersistence;
	pushConstants.lacunarity = params.perlinLacunarity;
	pushConstants.brightness = params.perlinBrightness;
	pushConstants.contrast = params.perlinContrast;
	pushConstants.seed = params.perlinSeed;
	pushConstants.textureSize = width;
	pushConstants.time = params.perlinTime;
    // Primary/secondary layer indices refer to layers in the texture array manager
	pushConstants.primaryLayer = static_cast<uint32_t>(params.primaryTextureIdx);
	pushConstants.secondaryLayer = static_cast<uint32_t>(params.secondaryTextureIdx);
    // Destination layer to write into (if using array layers)
    pushConstants.targetLayer = static_cast<uint32_t>(params.targetLayer);


	// Require a TextureArrayManager for array-based generation
	if (!textureArrayManager) throw std::runtime_error("TextureMixer requires a TextureArrayManager for array-based generation");

	VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

	// Transition the three target images (or the selected array layer) to GENERAL for compute write
	VkImageMemoryBarrier barriers[3]{};
	for (int i = 0; i < 3; ++i) {
		barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barriers[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barriers[i].subresourceRange.baseMipLevel = 0;
		barriers[i].subresourceRange.levelCount = 1;
		// default to layer 0 for editable textures; may be overridden below when writing into arrays
		barriers[i].subresourceRange.baseArrayLayer = 0;
		barriers[i].subresourceRange.layerCount = 1;
		barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	}

	// If a TextureArrayManager is present and a valid target layer was specified,
	// update the storage-image descriptors to point at the per-layer 2D views
	// so the compute shader can write directly into the array layer. Otherwise
	// write into the editable 2D textures as before.
	bool useArrayLayer = false;
	uint32_t targetLayer = 0;
	if (textureArrayManager) {
		targetLayer = params.targetLayer;
		if (targetLayer < textureArrayManager->layerAmount) {
			useArrayLayer = true;
			// Ensure per-layer views exist for preview purposes
			textureArrayManager->getImTexture(targetLayer, 0);
			textureArrayManager->getImTexture(targetLayer, 1);
			textureArrayManager->getImTexture(targetLayer, 2);

			// Transition array image layers to GENERAL for compute write
			barriers[0].image = textureArrayManager->albedoArray.image;
			barriers[0].subresourceRange.baseArrayLayer = targetLayer;
			barriers[1].image = textureArrayManager->normalArray.image;
			barriers[1].subresourceRange.baseArrayLayer = targetLayer;
			barriers[2].image = textureArrayManager->bumpArray.image;
			barriers[2].subresourceRange.baseArrayLayer = targetLayer;

			// If the target array layer hasn't been initialized yet it may be in UNDEFINED
			for (int i = 0; i < 3; ++i) {
				if (textureArrayManager->isLayerInitialized(targetLayer)) barriers[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				else barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
				barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
				barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			}
		}
	}

	if (!useArrayLayer) {
		throw std::runtime_error("TextureMixer: invalid target layer or missing TextureArrayManager");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		3, barriers
	);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);

	vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PerlinPushConstants), &pushConstants);

	uint32_t groupCountX = (width + 15) / 16;
	uint32_t groupCountY = (height + 15) / 16;

	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
	if (useArrayLayer && textureArrayManager) {
		// After compute wrote directly into the array layer, transition it back to SHADER_READ_ONLY_OPTIMAL
		VkImageMemoryBarrier postBarriers[3]{};
		for (int i = 0; i < 3; ++i) {
			postBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			postBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			postBarriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			postBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			postBarriers[i].subresourceRange.baseMipLevel = 0;
			postBarriers[i].subresourceRange.levelCount = 1;
			postBarriers[i].subresourceRange.baseArrayLayer = targetLayer;
			postBarriers[i].subresourceRange.layerCount = 1;
			postBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			postBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		postBarriers[0].image = textureArrayManager->albedoArray.image;
		postBarriers[1].image = textureArrayManager->normalArray.image;
		postBarriers[2].image = textureArrayManager->bumpArray.image;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			3, postBarriers
		);




		// Prepare base level for mipmap generation and flush commands
		VkImageMemoryBarrier prepBarriers[3]{};
		for (int i = 0; i < 3; ++i) {
			prepBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			prepBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			prepBarriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			prepBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			prepBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			prepBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			prepBarriers[i].subresourceRange.baseMipLevel = 0;
			prepBarriers[i].subresourceRange.levelCount = 1;
			prepBarriers[i].subresourceRange.baseArrayLayer = targetLayer;
			prepBarriers[i].subresourceRange.layerCount = 1;
			prepBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			prepBarriers[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		}
		prepBarriers[0].image = textureArrayManager->albedoArray.image;
		prepBarriers[1].image = textureArrayManager->normalArray.image;
		prepBarriers[2].image = textureArrayManager->bumpArray.image;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			3, prepBarriers
		);

		// End the command buffer so we can run mipmap generation commands
		app->endSingleTimeCommands(commandBuffer);

		// Generate mipmaps for the target layer on each array (if they have multiple mip levels)
		if (textureArrayManager->albedoArray.mipLevels > 1) {
app->generateMipmaps(textureArrayManager->albedoArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->albedoArray.mipLevels, 1, targetLayer);
			}
			if (textureArrayManager->normalArray.mipLevels > 1) {
				app->generateMipmaps(textureArrayManager->normalArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->normalArray.mipLevels, 1, targetLayer);
			}
			if (textureArrayManager->bumpArray.mipLevels > 1) {
				app->generateMipmaps(textureArrayManager->bumpArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->bumpArray.mipLevels, 1, targetLayer);
		}

		// Mark the layer initialized after the write
		textureArrayManager->setLayerInitialized(targetLayer, true);

		// Additional debug: print sampler handles and the array view used for sampling
		printf("[TextureMixer] albedoSampler=%p albedoArray.view=%p\n",
			(void*)textureArrayManager->albedoSampler, (void*)textureArrayManager->albedoArray.view);
		printf("[TextureMixer] normalSampler=%p normalArray.view=%p\n",
			(void*)textureArrayManager->normalSampler, (void*)textureArrayManager->normalArray.view);
		printf("[TextureMixer] bumpSampler=%p bumpArray.view=%p\n",
			(void*)textureArrayManager->bumpSampler, (void*)textureArrayManager->bumpArray.view);
	} else {
		// Transition images back to SHADER_READ_ONLY_OPTIMAL
		for (int i = 0; i < 3; ++i) {
			barriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			3, barriers
		);

		app->endSingleTimeCommands(commandBuffer);
	}

	printf("[TextureMixer] generatePerlinNoise: useArrayLayer=%d targetLayer=%u primary=%u secondary=%u\n",
		useArrayLayer ? 1 : 0, targetLayer, pushConstants.primaryLayer, pushConstants.secondaryLayer);

	// Debug: print the image views used for storage images
	if (useArrayLayer && textureArrayManager) {
		printf("[TextureMixer] albedoArray.image=%p albedoLayerView=%p\n", (void*)textureArrayManager->albedoArray.image, (void*)textureArrayManager->albedoLayerViews[targetLayer]);
		printf("[TextureMixer] normalArray.image=%p normalLayerView=%p\n", (void*)textureArrayManager->normalArray.image, (void*)textureArrayManager->normalLayerViews[targetLayer]);
		printf("[TextureMixer] bumpArray.image=%p bumpLayerView=%p\n", (void*)textureArrayManager->bumpArray.image, (void*)textureArrayManager->bumpLayerViews[targetLayer]);
	}

	printf("Perlin noise generation complete!\n");

	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}
