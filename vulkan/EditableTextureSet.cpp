#include "EditableTextureSet.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include "PerlinPushConstants.hpp"
#include "TextureArrayManager.hpp"
#include <algorithm>
#include <stdexcept>

EditableTextureSet::EditableTextureSet() {}

EditableTexture& EditableTextureSet::getAlbedo() { return albedo; }
EditableTexture& EditableTextureSet::getNormal() { return normal; }
EditableTexture& EditableTextureSet::getBump() { return bump; }

const EditableTexture& EditableTextureSet::getAlbedo() const { return albedo; }
const EditableTexture& EditableTextureSet::getNormal() const { return normal; }
const EditableTexture& EditableTextureSet::getBump() const { return bump; }

void EditableTextureSet::init(VulkanApp* app, uint32_t width, uint32_t height, const char* windowName, TextureArrayManager* textureArrayManager) {
	this->app = app;
	this->textureArrayManager = textureArrayManager;

	albedo.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Albedo");
	normal.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Normal");
	bump.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Bump");

	// Create compute pipeline and descriptor sets so we can generate textures on demand
	createComputePipeline();
	printf("[EditableTextureSet] Compute pipeline created for editable textures\n");
}

// Backwards-compatible overload: callers that don't pass a TextureArrayManager
void EditableTextureSet::init(VulkanApp* app, uint32_t width, uint32_t height, const char* windowName) {
	init(app, width, height, windowName, nullptr);
}

// setTextureManager removed — EditableTextureSet creates its own compute sampler
// and compute pipeline during init

void EditableTextureSet::setOnTextureGenerated(std::function<void()> callback) {
	onTextureGeneratedCallback = callback;
}

void EditableTextureSet::generateInitialTextures() {
	printf("Generating initial textures (Albedo, Normal, Bump)...\n");
	generatePerlinNoise(albedo);
	generatePerlinNoise(normal);
	generatePerlinNoise(bump);
}

void EditableTextureSet::cleanup() {
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


void EditableTextureSet::createComputePipeline() {
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
	poolSizes[0].descriptorCount = 3; // albedo, normal, bump
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = 3; // albedo array, normal array, bump array

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	poolInfo.maxSets = 3;

	if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor pool!");
	}

	// Create a single descriptor set that binds albedo/normal/bump storage images and samplers
	createTripleComputeDescriptorSet();
	// For backward compatibility, set individual handles to the triple set
	albedoComputeDescSet = normalComputeDescSet = bumpComputeDescSet = tripleComputeDescSet;
}

void EditableTextureSet::createTripleComputeDescriptorSet() {
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

VkDescriptorSet EditableTextureSet::getPreviewDescriptor(int map) {
	if (textureArrayManager && editableLayer != UINT32_MAX) {
		ImTextureID id = textureArrayManager->getImTexture(editableLayer, map);
		return (VkDescriptorSet)id;
	}
	// Fallback to editable texture descriptor
	if (map == 0) return albedo.getImGuiDescriptorSet();
	if (map == 1) return normal.getImGuiDescriptorSet();
	return bump.getImGuiDescriptorSet();
}

void EditableTextureSet::createComputeDescriptorSet(EditableTexture& texture, VkDescriptorSet& descSet) {
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
		samplerInfo.imageView = textureArrayManager->albedoArray.view;
		samplerInfo.sampler = textureArrayManager->albedoSampler;
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

void EditableTextureSet::generatePerlinNoise(EditableTexture& texture) {
	// generate regardless of external texture lists

	// We now perform a single dispatch that writes all three images (albedo, normal, bump)
	VkDescriptorSet descSet = tripleComputeDescSet;
	if (descSet == VK_NULL_HANDLE) return;

	PerlinPushConstants pushConstants;
	pushConstants.scale = perlinScale;
	pushConstants.octaves = perlinOctaves;
	pushConstants.persistence = perlinPersistence;
	pushConstants.lacunarity = perlinLacunarity;
	pushConstants.brightness = perlinBrightness;
	pushConstants.contrast = perlinContrast;
	pushConstants.seed = perlinSeed;
	pushConstants.textureSize = texture.getWidth();
	pushConstants.time = perlinTime;
    // Primary/secondary layer indices refer to layers in the texture array manager
	pushConstants.primaryLayer = static_cast<uint32_t>(primaryTextureIdx);
	pushConstants.secondaryLayer = static_cast<uint32_t>(secondaryTextureIdx);


	VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

	// Transition the three target images to GENERAL for compute write
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
		barriers[i].subresourceRange.baseArrayLayer = 0;
		barriers[i].subresourceRange.layerCount = 1;
		barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	}
	barriers[0].image = albedo.getImage();
	barriers[1].image = normal.getImage();
	barriers[2].image = bump.getImage();

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

	uint32_t groupCountX = (texture.getWidth() + 15) / 16;
	uint32_t groupCountY = (texture.getHeight() + 15) / 16;

	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

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

	printf("Perlin noise generation complete!\n");

	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}

void EditableTextureSet::generatePerlinNoiseWithParams(EditableTexture& texture, float scale, float octaves, float persistence, float lacunarity, float brightness, float contrast, float time, uint32_t seed) {
	// Temporarily override internal parameters and invoke generator
	float oldScale = perlinScale;
	float oldOctaves = perlinOctaves;
	float oldPersistence = perlinPersistence;
	float oldLacunarity = perlinLacunarity;
	float oldBrightness = perlinBrightness;
	float oldContrast = perlinContrast;
	float oldTime = perlinTime;
	uint32_t oldSeed = perlinSeed;

	perlinScale = scale;
	perlinOctaves = octaves;
	perlinPersistence = persistence;
	perlinLacunarity = lacunarity;
	perlinBrightness = brightness;
	perlinContrast = contrast;
	perlinTime = time;
	perlinSeed = seed;

	generatePerlinNoise(texture);

	// restore
	perlinScale = oldScale;
	perlinOctaves = oldOctaves;
	perlinPersistence = oldPersistence;
	perlinLacunarity = oldLacunarity;
	perlinBrightness = oldBrightness;
	perlinContrast = oldContrast;
	perlinTime = oldTime;
	perlinSeed = oldSeed;
}
