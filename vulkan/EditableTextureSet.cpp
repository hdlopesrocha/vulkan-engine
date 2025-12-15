#include "EditableTextureSet.hpp"
#include "FileReader.hpp"
#include <algorithm>
#include <stdexcept>

void EditableTextureSet::init(VulkanApp* app, uint32_t width, uint32_t height, const char* windowName) {
	this->app = app;
	this->title = windowName;

	albedo.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Albedo");
	normal.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Normal");
	bump.init(app, width, height, VK_FORMAT_R8G8B8A8_UNORM, "Bump");
}

void EditableTextureSet::setTextureManager(TextureManager* texMgr) {
	this->textureMgr = texMgr;
	createComputePipeline();
}

void EditableTextureSet::setOnTextureGenerated(std::function<void()> callback) {
	onTextureGeneratedCallback = callback;
}

void EditableTextureSet::generateInitialTextures() {
	if (!textureMgr || textureMgr->count() == 0) {
		printf("Cannot generate initial textures: No textures in TextureManager\n");
		return;
	}
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
}

void EditableTextureSet::render() {
	if (!ImGui::Begin(title.c_str(), &isOpen, ImGuiWindowFlags_None)) {
		ImGui::End();
		return;
	}

	if (ImGui::BeginTabBar("TextureTabBar")) {
		if (ImGui::BeginTabItem("Albedo")) {
			renderTextureTab(albedo);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Normal")) {
			renderTextureTab(normal);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Bump")) {
			renderTextureTab(bump);
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	ImGui::End();
}

void EditableTextureSet::renderTextureTab(EditableTexture& texture) {
	ImGui::Text("Size: %dx%d", texture.getWidth(), texture.getHeight());

	const char* formatName = "Unknown";
	if (texture.getBytesPerPixel() == 4) {
		formatName = "RGBA8";
	} else if (texture.getBytesPerPixel() == 1) {
		formatName = "R8";
	}
	ImGui::Text("Format: %s", formatName);

	if (texture.getDirty()) {
		ImGui::TextColored(ImVec4(1, 1, 0, 1), "Texture has unsaved changes");
		if (ImGui::Button("Upload to GPU")) {
			texture.updateGPU();
		}
	} else {
		ImGui::TextColored(ImVec4(0, 1, 0, 1), "Texture is up to date");
	}

	ImGui::Separator();

	ImGui::Text("Perlin Noise Generator");

	bool paramsChanged = false;

	if (textureMgr && textureMgr->count() > 0) {
		ImGui::Text("Texture Selection:");

		if (ImGui::Combo("Primary Texture", &primaryTextureIdx, [](void* data, int idx, const char** out_text) {
			static char buf[32];
			snprintf(buf, sizeof(buf), "Texture %d", idx);
			*out_text = buf;
			return true;
		}, nullptr, textureMgr->count())) {
			paramsChanged = true;
		}

		if (ImGui::Combo("Secondary Texture", &secondaryTextureIdx, [](void* data, int idx, const char** out_text) {
			static char buf[32];
			snprintf(buf, sizeof(buf), "Texture %d", idx);
			*out_text = buf;
			return true;
		}, nullptr, textureMgr->count())) {
			paramsChanged = true;
		}
	} else {
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "No textures loaded in TextureManager");
	}

	ImGui::Separator();
	ImGui::Text("Noise Parameters");

	int scaleInt = (int)perlinScale;
	if (ImGui::SliderInt("Scale", &scaleInt, 1, 32)) {
		perlinScale = (float)scaleInt;
		paramsChanged = true;
	}
	if (ImGui::SliderFloat("Octaves", &perlinOctaves, 1.0f, 8.0f)) {
		paramsChanged = true;
	}
	if (ImGui::SliderFloat("Persistence", &perlinPersistence, 0.0f, 1.0f)) {
		paramsChanged = true;
	}
	if (ImGui::SliderFloat("Lacunarity", &perlinLacunarity, 1.0f, 4.0f)) {
		paramsChanged = true;
	}
	if (ImGui::SliderFloat("Time", &perlinTime, 0.0f, 100.0f)) {
		paramsChanged = true;
	}

	ImGui::Separator();
	ImGui::Text("Adjustments");

	if (ImGui::SliderFloat("Brightness", &perlinBrightness, -1.0f, 1.0f)) {
		paramsChanged = true;
	}
	if (ImGui::SliderFloat("Contrast", &perlinContrast, 0.0f, 5.0f)) {
		paramsChanged = true;
	}

	if (paramsChanged) {
		generatePerlinNoise(albedo);
		generatePerlinNoise(normal);
		generatePerlinNoise(bump);
		prevPerlinScale = perlinScale;
		prevPerlinOctaves = perlinOctaves;
		prevPerlinPersistence = perlinPersistence;
		prevPerlinLacunarity = perlinLacunarity;
		prevPerlinBrightness = perlinBrightness;
		prevPerlinContrast = perlinContrast;
		prevPerlinTime = perlinTime;
		prevPrimaryTextureIdx = primaryTextureIdx;
		prevSecondaryTextureIdx = secondaryTextureIdx;
	}

	if (ImGui::Button("Generate Perlin Noise")) {
		generatePerlinNoise(texture);
	}
	ImGui::SameLine();
	if (ImGui::Button("Randomize Seed")) {
		std::random_device rd;
		perlinSeed = rd();
		generatePerlinNoise(texture);
	}
	ImGui::SameLine();
	if (ImGui::Button("Generate All")) {
		generatePerlinNoise(albedo);
		generatePerlinNoise(normal);
		generatePerlinNoise(bump);
		prevPerlinScale = perlinScale;
		prevPerlinOctaves = perlinOctaves;
		prevPerlinPersistence = perlinPersistence;
		prevPerlinLacunarity = perlinLacunarity;
		prevPerlinBrightness = perlinBrightness;
		prevPerlinContrast = perlinContrast;
		prevPerlinTime = perlinTime;
		prevPrimaryTextureIdx = primaryTextureIdx;
		prevSecondaryTextureIdx = secondaryTextureIdx;
	}

	ImGui::Separator();

	float previewSize = 512.0f;
	ImVec2 imageSize(previewSize, previewSize);
	ImGui::Text("Preview:");

	VkDescriptorSet descSet = texture.getImGuiDescriptorSet();
	if (descSet != VK_NULL_HANDLE) {
		ImGui::Image((ImTextureID)descSet, imageSize);
	} else {
		ImGui::Text("Texture preview not available");
	}
}

void EditableTextureSet::createComputePipeline() {
	VkDescriptorSetLayoutBinding bindings[3] = {};

	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	bindings[2].binding = 2;
	bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	bindings[2].descriptorCount = 1;
	bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 3;
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
	poolSizes[0].descriptorCount = 3;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = 6;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	poolInfo.maxSets = 3;

	if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor pool!");
	}

	createComputeDescriptorSet(albedo, albedoComputeDescSet);
	createComputeDescriptorSet(normal, normalComputeDescSet);
	createComputeDescriptorSet(bump, bumpComputeDescSet);
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

	vkUpdateDescriptorSets(app->getDevice(), 1, &descriptorWrite, 0, nullptr);
}

void EditableTextureSet::generatePerlinNoise(EditableTexture& texture) {
	if (!textureMgr || textureMgr->count() == 0) {
		printf("Cannot generate: No textures in TextureManager\n");
		return;
	}

	VkDescriptorSet descSet = VK_NULL_HANDLE;
	int textureType = 0;

	if (&texture == &albedo) {
		descSet = albedoComputeDescSet;
		textureType = 0;
	} else if (&texture == &normal) {
		descSet = normalComputeDescSet;
		textureType = 1;
	} else if (&texture == &bump) {
		descSet = bumpComputeDescSet;
		textureType = 2;
	}

	if (descSet == VK_NULL_HANDLE) {
		return;
	}

	int primIdx = std::min(primaryTextureIdx, (int)textureMgr->count() - 1);
	int secIdx = std::min(secondaryTextureIdx, (int)textureMgr->count() - 1);

	const auto& primaryTriple = textureMgr->getTriple(primIdx);
	const auto& secondaryTriple = textureMgr->getTriple(secIdx);

	VkDescriptorImageInfo primaryImageInfo{};
	VkDescriptorImageInfo secondaryImageInfo{};
	primaryImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	secondaryImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	switch(textureType) {
		case 0:
			primaryImageInfo.imageView = primaryTriple.albedo.view;
			primaryImageInfo.sampler = primaryTriple.albedoSampler;
			secondaryImageInfo.imageView = secondaryTriple.albedo.view;
			secondaryImageInfo.sampler = secondaryTriple.albedoSampler;
			break;
		case 1:
			primaryImageInfo.imageView = primaryTriple.normal.view;
			primaryImageInfo.sampler = primaryTriple.normalSampler;
			secondaryImageInfo.imageView = secondaryTriple.normal.view;
			secondaryImageInfo.sampler = secondaryTriple.normalSampler;
			break;
		case 2:
			primaryImageInfo.imageView = primaryTriple.height.view;
			primaryImageInfo.sampler = primaryTriple.heightSampler;
			secondaryImageInfo.imageView = secondaryTriple.height.view;
			secondaryImageInfo.sampler = secondaryTriple.heightSampler;
			break;
	}

	VkWriteDescriptorSet descriptorWrites[2] = {};

	descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[0].dstSet = descSet;
	descriptorWrites[0].dstBinding = 1;
	descriptorWrites[0].dstArrayElement = 0;
	descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrites[0].descriptorCount = 1;
	descriptorWrites[0].pImageInfo = &primaryImageInfo;

	descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[1].dstSet = descSet;
	descriptorWrites[1].dstBinding = 2;
	descriptorWrites[1].dstArrayElement = 0;
	descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrites[1].descriptorCount = 1;
	descriptorWrites[1].pImageInfo = &secondaryImageInfo;

	vkUpdateDescriptorSets(app->getDevice(), 2, descriptorWrites, 0, nullptr);

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

	const char* typeNames[] = {"Albedo", "Normal", "Bump"};

	printf("Generating %s Perlin noise: scale=%.2f, octaves=%.0f, persistence=%.2f, lacunarity=%.2f, brightness=%.2f, contrast=%.2f, seed=%u\n",
		   typeNames[textureType], pushConstants.scale, pushConstants.octaves, pushConstants.persistence, pushConstants.lacunarity,
		   pushConstants.brightness, pushConstants.contrast, pushConstants.seed);
	printf("Primary texture: %d, Secondary texture: %d\n", primIdx, secIdx);
	printf("Texture size: %dx%d, dispatch groups: %dx%d\n",
		   texture.getWidth(), texture.getHeight(),
		   (texture.getWidth() + 15) / 16, (texture.getHeight() + 15) / 16);

	VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = texture.getImage();
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);

	vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PerlinPushConstants), &pushConstants);

	uint32_t groupCountX = (texture.getWidth() + 15) / 16;
	uint32_t groupCountY = (texture.getHeight() + 15) / 16;
	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

	barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	app->endSingleTimeCommands(commandBuffer);

	printf("Perlin noise generation complete!\n");

	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}
