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

TextureMixer::TextureMixer() {}

EditableTexture& TextureMixer::getAlbedo() { return albedo; }
EditableTexture& TextureMixer::getNormal() { return normal; }
EditableTexture& TextureMixer::getBump() { return bump; }

const EditableTexture& TextureMixer::getAlbedo() const { return albedo; }
const EditableTexture& TextureMixer::getNormal() const { return normal; }
const EditableTexture& TextureMixer::getBump() const { return bump; }

void TextureMixer::init(VulkanApp* app, uint32_t width, uint32_t height, TextureArrayManager* textureArrayManager) {
	this->app = app;
	this->textureArrayManager = textureArrayManager;
	this->width = width;
	this->height = height;
	albedo.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Albedo");
	normal.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Normal");
	bump.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Bump");

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
	albedo.cleanup();
	normal.cleanup();
	bump.cleanup();

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
	createComputeDescriptorSet(albedo, albedoComputeDescSet, 0);
	createComputeDescriptorSet(normal, normalComputeDescSet, 1);
	createComputeDescriptorSet(bump, bumpComputeDescSet, 2);
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
	albedoImageInfo.imageView = albedo.getView();
	albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo normalImageInfo{};
	normalImageInfo.imageView = normal.getView();
	normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo bumpImageInfo{};
	bumpImageInfo.imageView = bump.getView();
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
		// Fallback to individual editable textures with computeSampler
		if (computeSampler == VK_NULL_HANDLE) computeSampler = app->createTextureSampler(1);
		albedoSamplerInfo.imageView = albedo.getView(); albedoSamplerInfo.sampler = computeSampler; albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		normalSamplerInfo.imageView = normal.getView(); normalSamplerInfo.sampler = computeSampler; normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		bumpSamplerInfo.imageView = bump.getView(); bumpSamplerInfo.sampler = computeSampler; bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
	if (map == 0) return albedo.getImGuiDescriptorSet();
	if (map == 1) return normal.getImGuiDescriptorSet();
	return bump.getImGuiDescriptorSet();
}

VkDescriptorSet TextureMixer::getPreviewDescriptor(int map, uint32_t layer) {
	if (textureArrayManager && layer < textureArrayManager->layerAmount) {
		ImTextureID id = textureArrayManager->getImTexture(layer, map);
		return (VkDescriptorSet)id;
	}
	// Fallback to default behavior
	return getPreviewDescriptor(map);
}

void TextureMixer::createComputeDescriptorSet(EditableTexture& texture, VkDescriptorSet& descSet, int map) {
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = computeDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &computeDescriptorSetLayout;

	if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &descSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate compute descriptor set!");
	}

	VkDescriptorImageInfo imageInfo{};
	imageInfo.imageView = texture.getView();
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
		// Fallback: bind the editable texture itself as the sampler input
		samplerInfo.imageView = texture.getView();
		samplerInfo.sampler = computeSampler;
		samplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	VkWriteDescriptorSet samplerWrite1{}; samplerWrite1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite1.dstSet = descSet; samplerWrite1.dstBinding = 1; samplerWrite1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite1.descriptorCount = 1; samplerWrite1.pImageInfo = &samplerInfo;
	VkWriteDescriptorSet samplerWrite2{}; samplerWrite2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; samplerWrite2.dstSet = descSet; samplerWrite2.dstBinding = 2; samplerWrite2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; samplerWrite2.descriptorCount = 1; samplerWrite2.pImageInfo = &samplerInfo;

	VkWriteDescriptorSet writes[3] = { descriptorWrite, samplerWrite1, samplerWrite2 };
	vkUpdateDescriptorSets(app->getDevice(), 3, writes, 0, nullptr);
}

void TextureMixer::generatePerlinNoise(MixerParameters &params) {
	// generate regardless of external texture lists

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
	// and transition the corresponding array image/layer instead of the editable textures.
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

			// Instead of writing directly into the array layer via storage-image views,
			// we generate into the editable 2D textures (default storage images) and
			// then copy those editable images into the specified array layer.
			// Prepare barriers so the editable textures are transitioned to GENERAL
			// for compute writes (they are already the default storage-image targets).
			barriers[0].image = albedo.getImage();
			barriers[1].image = normal.getImage();
			barriers[2].image = bump.getImage();
			// baseArrayLayer remains 0 for these 2D editable images
		}
	}

	if (!useArrayLayer) {
		barriers[0].image = albedo.getImage();
		barriers[1].image = normal.getImage();
		barriers[2].image = bump.getImage();
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
		// After compute wrote into the editable 2D images, copy each editable image
		// into the corresponding array image layer.

		// Transition editable images (src) to TRANSFER_SRC_OPTIMAL and
		// array images (dst, specific layer) to TRANSFER_DST_OPTIMAL.
		VkImageMemoryBarrier copyBarriers[6]{};
		// editable -> TRANSFER_SRC
		for (int i = 0; i < 3; ++i) {
			copyBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			copyBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			copyBarriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			copyBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copyBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copyBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyBarriers[i].subresourceRange.baseMipLevel = 0;
			copyBarriers[i].subresourceRange.levelCount = 1;
			copyBarriers[i].subresourceRange.baseArrayLayer = 0;
			copyBarriers[i].subresourceRange.layerCount = 1;
			copyBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			copyBarriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		}

		// array layer -> TRANSFER_DST
		copyBarriers[0].image = albedo.getImage();
		copyBarriers[1].image = normal.getImage();
		copyBarriers[2].image = bump.getImage();

		copyBarriers[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		// If the target array layer hasn't been initialized yet it may be in UNDEFINED
		if (textureArrayManager->isLayerInitialized(targetLayer)) copyBarriers[3].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		else copyBarriers[3].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		copyBarriers[3].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		copyBarriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copyBarriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		copyBarriers[3].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyBarriers[3].subresourceRange.baseMipLevel = 0;
		copyBarriers[3].subresourceRange.levelCount = 1;
		copyBarriers[3].subresourceRange.baseArrayLayer = targetLayer;
		copyBarriers[3].subresourceRange.layerCount = 1;
		copyBarriers[3].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		copyBarriers[3].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		copyBarriers[3].image = textureArrayManager->albedoArray.image;

		copyBarriers[4] = copyBarriers[3];
		copyBarriers[4].image = textureArrayManager->normalArray.image;

		copyBarriers[5] = copyBarriers[3];
		copyBarriers[5].image = textureArrayManager->bumpArray.image;

		// Mark the layer initialized after the copy will complete
		textureArrayManager->setLayerInitialized(targetLayer, true);

		printf("[TextureMixer] copy: srcImages albedo=%p normal=%p bump=%p\n", (void*)albedo.getImage(), (void*)normal.getImage(), (void*)bump.getImage());
		printf("[TextureMixer] copy: dstArrayImages albedo=%p normal=%p bump=%p targetLayer=%u\n",
			(void*)textureArrayManager->albedoArray.image, (void*)textureArrayManager->normalArray.image, (void*)textureArrayManager->bumpArray.image, targetLayer);

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			6, copyBarriers
		);

		// Copy regions
		VkImageCopy copyRegion{};
		copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.srcSubresource.mipLevel = 0;
		copyRegion.srcSubresource.baseArrayLayer = 0;
		copyRegion.srcSubresource.layerCount = 1;
		copyRegion.srcOffset = {0,0,0};
		copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.dstSubresource.mipLevel = 0;
		copyRegion.dstSubresource.baseArrayLayer = targetLayer;
		copyRegion.dstSubresource.layerCount = 1;
		copyRegion.dstOffset = {0,0,0};
		copyRegion.extent = { (uint32_t)width, (uint32_t)height, 1 };

		vkCmdCopyImage(commandBuffer,
			albedo.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			textureArrayManager->albedoArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copyRegion);

		vkCmdCopyImage(commandBuffer,
			normal.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			textureArrayManager->normalArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copyRegion);

		vkCmdCopyImage(commandBuffer,
			bump.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			textureArrayManager->bumpArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copyRegion);

		// Transition array layers and editable images back to SHADER_READ_ONLY_OPTIMAL
		VkImageMemoryBarrier postBarriers[6]{};

		// editable images: TRANSFER_SRC_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
		for (int i = 0; i < 3; ++i) {
			postBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			postBarriers[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			postBarriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			postBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			postBarriers[i].subresourceRange.baseMipLevel = 0;
			postBarriers[i].subresourceRange.levelCount = 1;
			postBarriers[i].subresourceRange.baseArrayLayer = 0;
			postBarriers[i].subresourceRange.layerCount = 1;
			postBarriers[i].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			postBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		postBarriers[0].image = albedo.getImage();
		postBarriers[1].image = normal.getImage();
		postBarriers[2].image = bump.getImage();

		// array layers: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
		for (int i = 0; i < 3; ++i) {
			postBarriers[3 + i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			postBarriers[3 + i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			postBarriers[3 + i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			postBarriers[3 + i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[3 + i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			postBarriers[3 + i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			postBarriers[3 + i].subresourceRange.baseMipLevel = 0;
			postBarriers[3 + i].subresourceRange.levelCount = 1;
			postBarriers[3 + i].subresourceRange.baseArrayLayer = targetLayer;
			postBarriers[3 + i].subresourceRange.layerCount = 1;
			postBarriers[3 + i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			postBarriers[3 + i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}
		postBarriers[3].image = textureArrayManager->albedoArray.image;
		postBarriers[4].image = textureArrayManager->normalArray.image;
		postBarriers[5].image = textureArrayManager->bumpArray.image;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			6, postBarriers
		);

		app->endSingleTimeCommands(commandBuffer);
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
	} else {
		printf("[TextureMixer] editable images: albedo=%p normal=%p bump=%p\n", (void*)albedo.getImage(), (void*)normal.getImage(), (void*)bump.getImage());
	}

	printf("Perlin noise generation complete!\n");

	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}

void TextureMixer::generatePerlinNoiseWithParams(MixerParameters &params) {
	// Update internal size if caller provided different dimensions
	generatePerlinNoise(params);
}

void TextureMixer::generatePerlinNoiseForMap(MixerParameters &params, int map) {
	if (map < 0 || map > 2) return;
	// For simplicity use the unified triple-dispatch path which generates all maps.
	generatePerlinNoise(params);
}
