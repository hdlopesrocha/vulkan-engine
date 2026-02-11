#include "TextureMixer.hpp"
#include "VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include "PerlinPushConstants.hpp"
#include "TextureArrayManager.hpp"
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <tuple>
#include <string>

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

// Global instance pointer (set in init)
static TextureMixer* g_texture_mixer_instance = nullptr;
TextureMixer* TextureMixer::getGlobalInstance() { return g_texture_mixer_instance; }

void TextureMixer::init(VulkanApp* app, uint32_t width, uint32_t height, TextureArrayManager* textureArrayManager) {
	this->app = app;
	this->textureArrayManager = textureArrayManager;
	this->width = width;
	this->height = height;
	// Register global instance so other systems can query/wait on layer generations
	g_texture_mixer_instance = this;

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
	if (!textureArrayManager || textureArrayManager->layerAmount == 0) {
		std::lock_guard<std::mutex> lk(logsMutex);
		logs.emplace_back("Skipping generateInitialTextures: no texture arrays available");
		fprintf(stderr, "[TextureMixer] Skipping generateInitialTextures: no texture arrays available\n");
		return;
	}
	printf("Enqueuing initial textures for generation (Albedo, Normal, Bump)...\n");
	for (auto &param : mixerParams) {
		fprintf(stderr, "[TextureMixer] generateInitialTextures: enqueueing generation for layer=%zu\n", param.targetLayer);
		try {
			enqueueGenerate(param);
		} catch (const std::exception &e) {
			std::lock_guard<std::mutex> lk(logsMutex);
			char buf[256];
			snprintf(buf, sizeof(buf), "generateInitialTextures: enqueue failed for layer=%zu reason=%s", param.targetLayer, e.what());
			logs.emplace_back(buf);
			fprintf(stderr, "[TextureMixer] generateInitialTextures: enqueue failed for layer=%zu reason=%s\n", param.targetLayer, e.what());
		}
	}
}


// Queue a generation request from UI thread; will be flushed synchronously from the main update loop
void TextureMixer::enqueueGenerate(const MixerParameters &params, int map) {
	std::lock_guard<std::mutex> lk(pendingRequestsMutex);
	// Coalesce requests for the same target layer and map: replace older request if present
	bool replaced = false;
	for (auto &r : pendingRequests) {
		if (r.first.targetLayer == params.targetLayer && r.second == map) {
			r.first = params; // replace
			replaced = true;
			break;
		}
	}
	if (!replaced) pendingRequests.emplace_back(params, map);
	// Log the enqueue or replacement
	{
		std::lock_guard<std::mutex> lk2(logsMutex);
		char buf[128];
		snprintf(buf, sizeof(buf), "%s generation: layer=%zu map=%d", replaced ? "Replaced" : "Enqueued", params.targetLayer, map);
		logs.emplace_back(buf);
	}
}

// Flush pending requests synchronously; intended to be called from main update() before frame command buffers are recorded
void TextureMixer::flushPendingRequests() {
	std::vector<std::pair<MixerParameters,int>> tasks;
	{
		std::lock_guard<std::mutex> lk(pendingRequestsMutex);
		tasks.swap(pendingRequests);
	}
	for (auto &t : tasks) {
		// For lower-latency, submit generation asynchronously and track fences
		try {
			generatePerlinNoise(const_cast<MixerParameters&>(t.first), t.second);
		} catch (const std::exception &e) {
			std::lock_guard<std::mutex> lkll(logsMutex);
			char buf[256];
			snprintf(buf, sizeof(buf), "generate Perlin failed: layer=%zu map=%d reason=%s", t.first.targetLayer, t.second, e.what());
			logs.emplace_back(buf);
			fprintf(stderr, "[TextureMixer] generatePerlinNoise failed: %s\n", e.what());
		}
	}
}

void TextureMixer::pollPendingGenerations() {
	// Pull any completed fences and promote their logs (check fences BEFORE letting VulkanApp destroy them)

	std::vector<std::tuple<VkFence, uint32_t>> completed;
	{
		std::lock_guard<std::mutex> lk(pendingFencesMutex);
		for (auto it = pendingFences.begin(); it != pendingFences.end(); ) {
			VkFence f = std::get<0>(*it);
			uint32_t layer = std::get<1>(*it);
			if (!app) { ++it; continue; }
			// If VulkanApp doesn't know about this fence any more, treat it as completed (it was cleaned up elsewhere)
			if (!app || !app->isFencePending(f)) {
				completed.push_back(*it);
				it = pendingFences.erase(it);
				continue;
			}
			VkResult st = vkGetFenceStatus(app->getDevice(), f);
			if (st == VK_SUCCESS) {
				// generation complete
				completed.push_back(*it);
				it = pendingFences.erase(it);
			} else if (st == VK_ERROR_DEVICE_LOST) {
				it = pendingFences.erase(it);
			} else {
				++it;
			}
		}
	}

	// Let VulkanApp process and cleanup any pending command buffers/fences now that we've recorded completed ones
	if (app) app->processPendingCommandBuffers();

	for (auto &c : completed) {
		uint32_t layer = std::get<1>(c);
		// Mark layer initialized so previews show up
		if (textureArrayManager) {
			textureArrayManager->setLayerInitialized(layer, true);
			// After generation completes, ensure tracked layout is SHADER_READ_ONLY_OPTIMAL for all maps
			textureArrayManager->setLayerLayout(0, layer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			textureArrayManager->setLayerLayout(1, layer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			textureArrayManager->setLayerLayout(2, layer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		if (onTextureGeneratedCallback) onTextureGeneratedCallback();
		{
			std::lock_guard<std::mutex> lkll(logsMutex);
			char buf[128];
			snprintf(buf, sizeof(buf), "Generation complete: layer=%u", layer);
			logs.emplace_back(buf);
		}
	}

	// also append a simple summary log line for diagnostics
	{
		std::lock_guard<std::mutex> lkll(logsMutex);
		char buf[128];
		snprintf(buf, sizeof(buf), "Pending: requests=%zu fences=%zu", pendingRequests.size(), pendingFences.size());
		logs.emplace_back(buf);
	}
}

size_t TextureMixer::getPendingGenerationCount() {
	std::lock_guard<std::mutex> lk1(pendingRequestsMutex);
	std::lock_guard<std::mutex> lk2(pendingFencesMutex);
	return pendingRequests.size() + pendingFences.size();
}

std::vector<std::string> TextureMixer::consumeLogs() {
	std::lock_guard<std::mutex> lk(logsMutex);
	auto out = logs;
	logs.clear();
	return out;
}


bool TextureMixer::isLayerGenerationPending(uint32_t layer) {
	std::lock_guard<std::mutex> lk(pendingFencesMutex);
	for (auto &t : pendingFences) {
		if (std::get<1>(t) == layer) return true;
	}
	return false;
}

bool TextureMixer::waitForLayerGeneration(uint32_t layer, uint64_t timeoutNs) {
	TextureMixer* g = getGlobalInstance();
	if (!g || !g->app) return false;
	std::vector<VkFence> fences;
	{
		std::lock_guard<std::mutex> lk(pendingFencesMutex);
		for (auto &t : pendingFences) {
			if (std::get<1>(t) == layer) fences.push_back(std::get<0>(t));
		}
	}
	if (fences.empty()) return false;
	VkResult r = vkWaitForFences(g->app->getDevice(), static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, timeoutNs);
	if (r == VK_SUCCESS) {
		// process pending generations so state advances
		if (g->app) g->app->processPendingCommandBuffers();
		g->pollPendingGenerations();
		return true;
	}
	return false;
}

void TextureMixer::cleanup() {
	// Clear global instance pointer
	if (g_texture_mixer_instance == this) g_texture_mixer_instance = nullptr;

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

	// clear log buffer
	{
		std::lock_guard<std::mutex> lk(logsMutex);
		logs.clear();
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
	poolSizes[0].descriptorCount = 20; // provide extra slack
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

	if (computeDescriptorSetLayout == VK_NULL_HANDLE) {
		fprintf(stderr, "[TextureMixer::createTripleComputeDescriptorSet] ERROR: computeDescriptorSetLayout is VK_NULL_HANDLE\n");
		throw std::runtime_error("TextureMixer: computeDescriptorSetLayout is VK_NULL_HANDLE");
	}
	if (computeDescriptorPool == VK_NULL_HANDLE) {
		fprintf(stderr, "[TextureMixer::createTripleComputeDescriptorSet] ERROR: computeDescriptorPool is VK_NULL_HANDLE\n");
		throw std::runtime_error("TextureMixer: computeDescriptorPool is VK_NULL_HANDLE");
	}
	if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &tripleComputeDescSet) != VK_SUCCESS) {
		fprintf(stderr, "[TextureMixer::createTripleComputeDescriptorSet] vkAllocateDescriptorSets failed\n");
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
		// Use GENERAL layout because during compute we transition all layers to GENERAL
		albedoSamplerInfo.imageView = textureArrayManager->albedoArray.view;
		albedoSamplerInfo.sampler = textureArrayManager->albedoSampler;
		albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		normalSamplerInfo.imageView = textureArrayManager->normalArray.view;
		normalSamplerInfo.sampler = textureArrayManager->normalSampler;
		normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		bumpSamplerInfo.imageView = textureArrayManager->bumpArray.view;
		bumpSamplerInfo.sampler = textureArrayManager->bumpSampler;
		bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	} else {
		// No TextureArrayManager: leave sampler imageViews null (not supported)
		albedoSamplerInfo.imageView = VK_NULL_HANDLE; albedoSamplerInfo.sampler = VK_NULL_HANDLE; albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		normalSamplerInfo.imageView = VK_NULL_HANDLE; normalSamplerInfo.sampler = VK_NULL_HANDLE; normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		bumpSamplerInfo.imageView = VK_NULL_HANDLE; bumpSamplerInfo.sampler = VK_NULL_HANDLE; bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	std::vector<VkWriteDescriptorSet> writes;

	auto addStorageImage = [&](uint32_t binding, VkDescriptorImageInfo &info){
		if (info.imageView == VK_NULL_HANDLE) {
			fprintf(stderr, "[TextureMixer] Skipping storage image binding %u: imageView=%p\n", binding, (void*)info.imageView);
			return;
		}
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = tripleComputeDescSet;
		w.dstBinding = binding;
		w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		w.descriptorCount = 1;
		w.pImageInfo = &info;
		writes.push_back(w);
	};

	auto addCombinedSampler = [&](uint32_t binding, VkDescriptorImageInfo &info){
		if (info.imageView == VK_NULL_HANDLE || info.sampler == VK_NULL_HANDLE) {
			fprintf(stderr, "[TextureMixer] Skipping sampler binding %u: imageView=%p sampler=%p\n", binding, (void*)info.imageView, (void*)info.sampler);
			return;
		}
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = tripleComputeDescSet;
		w.dstBinding = binding;
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.descriptorCount = 1;
		w.pImageInfo = &info;
		writes.push_back(w);
	};

	addStorageImage(0, albedoImageInfo);
	addCombinedSampler(1, albedoSamplerInfo);
	addCombinedSampler(2, normalSamplerInfo);
	addCombinedSampler(3, bumpSamplerInfo);
	addStorageImage(4, normalImageInfo);
	addStorageImage(5, bumpImageInfo);

	if (!writes.empty()) vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void TextureMixer::updateComputeDescriptorSets() {
	if (!app) return;
	if (!textureArrayManager) {
		std::lock_guard<std::mutex> lk(logsMutex);
		logs.emplace_back("updateComputeDescriptorSets: no TextureArrayManager attached");
		return;
	}

	VkDevice dev = app->getDevice();

	VkDescriptorImageInfo albedoImageInfo{};
	albedoImageInfo.imageView = textureArrayManager->albedoArray.view;
	albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo normalImageInfo{};
	normalImageInfo.imageView = textureArrayManager->normalArray.view;
	normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo bumpImageInfo{};
	bumpImageInfo.imageView = textureArrayManager->bumpArray.view;
	bumpImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorImageInfo albedoSamplerInfo{}; albedoSamplerInfo.imageView = textureArrayManager->albedoArray.view; albedoSamplerInfo.sampler = textureArrayManager->albedoSampler; albedoSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorImageInfo normalSamplerInfo{}; normalSamplerInfo.imageView = textureArrayManager->normalArray.view; normalSamplerInfo.sampler = textureArrayManager->normalSampler; normalSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorImageInfo bumpSamplerInfo{}; bumpSamplerInfo.imageView = textureArrayManager->bumpArray.view; bumpSamplerInfo.sampler = textureArrayManager->bumpSampler; bumpSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	std::vector<VkWriteDescriptorSet> writes;

	auto pushStorage = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorImageInfo *info) {
		if (set == VK_NULL_HANDLE || info->imageView == VK_NULL_HANDLE) return;
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = set;
		w.dstBinding = binding;
		w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		w.descriptorCount = 1;
		w.pImageInfo = info;
		writes.push_back(w);
	};
	auto pushSampler = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorImageInfo *info) {
		if (set == VK_NULL_HANDLE || info->imageView == VK_NULL_HANDLE || info->sampler == VK_NULL_HANDLE) return;
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = set;
		w.dstBinding = binding;
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.descriptorCount = 1;
		w.pImageInfo = info;
		writes.push_back(w);
	};

	// Update triple descriptor set (bind all storage images and samplers)
	if (tripleComputeDescSet != VK_NULL_HANDLE) {
		pushStorage(tripleComputeDescSet, 0, &albedoImageInfo);
		pushSampler(tripleComputeDescSet, 1, &albedoSamplerInfo);
		pushSampler(tripleComputeDescSet, 2, &normalSamplerInfo);
		pushSampler(tripleComputeDescSet, 3, &bumpSamplerInfo);
		pushStorage(tripleComputeDescSet, 4, &normalImageInfo);
		pushStorage(tripleComputeDescSet, 5, &bumpImageInfo);
	}

	// Update per-map descriptor sets if they were allocated
	if (albedoComputeDescSet != VK_NULL_HANDLE) {
		pushStorage(albedoComputeDescSet, 0, &albedoImageInfo);
		pushSampler(albedoComputeDescSet, 1, &albedoSamplerInfo);
		pushSampler(albedoComputeDescSet, 2, &albedoSamplerInfo); // keep existing behavior (bind same sampler to 2)
	}
	if (normalComputeDescSet != VK_NULL_HANDLE) {
		pushStorage(normalComputeDescSet, 0, &normalImageInfo);
		pushSampler(normalComputeDescSet, 1, &normalSamplerInfo);
		pushSampler(normalComputeDescSet, 2, &normalSamplerInfo);
	}
	if (bumpComputeDescSet != VK_NULL_HANDLE) {
		pushStorage(bumpComputeDescSet, 0, &bumpImageInfo);
		pushSampler(bumpComputeDescSet, 1, &bumpSamplerInfo);
		pushSampler(bumpComputeDescSet, 2, &bumpSamplerInfo);
	}

	if (!writes.empty()) {
		vkUpdateDescriptorSets(dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		std::lock_guard<std::mutex> lk(logsMutex);
		logs.emplace_back("updateComputeDescriptorSets: descriptor sets updated with texture arrays");
		fprintf(stderr, "[TextureMixer] updateComputeDescriptorSets: wrote %zu descriptors\n", writes.size());
	}
}

void TextureMixer::attachTextureArrayManager(TextureArrayManager* tam) {
	this->textureArrayManager = tam;
	fprintf(stderr, "[TextureMixer] attachTextureArrayManager called: tam=%p\n", (void*)tam);
	updateComputeDescriptorSets();
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

	if (computeDescriptorSetLayout == VK_NULL_HANDLE) {
		fprintf(stderr, "[TextureMixer::createComputeDescriptorSet] ERROR: computeDescriptorSetLayout is VK_NULL_HANDLE\n");
		throw std::runtime_error("TextureMixer: computeDescriptorSetLayout is VK_NULL_HANDLE");
	}
	if (computeDescriptorPool == VK_NULL_HANDLE) {
		fprintf(stderr, "[TextureMixer::createComputeDescriptorSet] ERROR: computeDescriptorPool is VK_NULL_HANDLE\n");
		throw std::runtime_error("TextureMixer: computeDescriptorPool is VK_NULL_HANDLE");
	}
	if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &descSet) != VK_SUCCESS) {
		fprintf(stderr, "[TextureMixer::createComputeDescriptorSet] vkAllocateDescriptorSets failed\n");
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
	// Use GENERAL layout since some layers may be in GENERAL during compute (target layer is transitioned)
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
		samplerInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	} else {
		samplerInfo.imageView = VK_NULL_HANDLE;
		samplerInfo.sampler = VK_NULL_HANDLE;
		samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	std::vector<VkWriteDescriptorSet> writes;
	if (imageInfo.imageView != VK_NULL_HANDLE) {
		VkWriteDescriptorSet w = descriptorWrite;
		writes.push_back(w);
	} else {
		fprintf(stderr, "[TextureMixer] Skipping createComputeDescriptorSet storage image for map %d: imageView=%p\n", map, (void*)imageInfo.imageView);
	}
	if (samplerInfo.imageView != VK_NULL_HANDLE && samplerInfo.sampler != VK_NULL_HANDLE) {
		VkWriteDescriptorSet s1{}; s1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; s1.dstSet = descSet; s1.dstBinding = 1; s1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; s1.descriptorCount = 1; s1.pImageInfo = &samplerInfo;
		VkWriteDescriptorSet s2{}; s2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; s2.dstSet = descSet; s2.dstBinding = 2; s2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; s2.descriptorCount = 1; s2.pImageInfo = &samplerInfo;
		writes.push_back(s1);
		writes.push_back(s2);
	} else {
		fprintf(stderr, "[TextureMixer] Skipping sampler bindings for map %d: imageView=%p sampler=%p\n", map, (void*)samplerInfo.imageView, (void*)samplerInfo.sampler);
	}
	if (!writes.empty()) vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void TextureMixer::generatePerlinNoise(MixerParameters &params, int map) {
	// log immediate sync generation requests too for diagnostics
	{
		std::lock_guard<std::mutex> lkll(logsMutex);
		char buf[128];
		snprintf(buf, sizeof(buf), "Immediate generate called: layer=%zu map=%d", params.targetLayer, map);
		logs.emplace_back(buf);
	}
	// generate regardless of external texture lists
	if(params.targetLayer == params.primaryTextureIdx || 
	   params.targetLayer == params.secondaryTextureIdx) {
		return; // avoid sampling and writing to the same layer
	}

	// Choose descriptor set and images to generate based on 'map' (-1 = all, 0=albedo,1=normal,2=bump)
	VkDescriptorSet descSet = VK_NULL_HANDLE;
	bool genA = false, genN = false, genB = false;
	// Prefer the triple descriptor set which binds all samplers and result storage images
	if (tripleComputeDescSet != VK_NULL_HANDLE) {
		// If map == -1 we generate all maps; otherwise only mark the requested one(s)
		descSet = tripleComputeDescSet;
		if (map == -1) { genA = genN = genB = true; }
		else if (map == 0) { genA = true; }
		else if (map == 1) { genN = true; }
		else if (map == 2) { genB = true; }
	} else {
		// Fallback to per-map descriptor sets (legacy behavior)
		if (map == -1) { descSet = tripleComputeDescSet; genA = genN = genB = true; }
		else if (map == 0) { descSet = albedoComputeDescSet; genA = true; }
		else if (map == 1) { descSet = normalComputeDescSet; genN = true; }
		else if (map == 2) { descSet = bumpComputeDescSet; genB = true; }
	}
	if (descSet == VK_NULL_HANDLE) {
		std::lock_guard<std::mutex> lkll(logsMutex);
		logs.emplace_back("No compute descriptor set available for Perlin generation");
		return;
	}

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

	// Build a list of image barriers only for the maps we will generate
	std::vector<VkImageMemoryBarrier> barriers;
	// mkBarrierLayer is used to transition only the target array layer (more precise, avoids whole-array races)
	auto mkBarrierLayer = [&](VkImage img, uint32_t baseArrayLayer, uint32_t layerCount, uint32_t mipLevels) {
		VkImageMemoryBarrier b{};
		b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // assume readable initially
		b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		b.subresourceRange.baseMipLevel = 0;
		b.subresourceRange.levelCount = mipLevels;
		b.subresourceRange.baseArrayLayer = baseArrayLayer;
		b.subresourceRange.layerCount = layerCount;
		b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		b.image = img;
		return b;
	};

	// If a TextureArrayManager is present and a valid target layer was specified,
	// ensure we will use array-layer views for storage writes and prepare
	// per-layer barriers (we only touch the requested layer to avoid races)
	bool useArrayLayer = false;
	uint32_t targetLayer = 0;
	if (textureArrayManager) {
		targetLayer = params.targetLayer;
		if (targetLayer < textureArrayManager->layerAmount) {
			useArrayLayer = true;
			// Ensure per-layer views exist for preview purposes
			if (genA) textureArrayManager->getImTexture(targetLayer, 0);
			if (genN) textureArrayManager->getImTexture(targetLayer, 1);
			if (genB) textureArrayManager->getImTexture(targetLayer, 2);

			// Before building barriers, ensure any prior generation or transfer that
			// touches this layer has completed. We call processPendingCommandBuffers
			// to advance completed fences, then wait briefly if the tracked layout
			// indicates a transfer-in-progress. This prevents using stale tracked
			// layouts as barrier.oldLayout.
			if (app) app->processPendingCommandBuffers();
			// If another generation for this layer is pending, wait for it.
			if (isLayerGenerationPending(targetLayer)) {
				waitForLayerGeneration(targetLayer);
			}
			// If tracked layout is TRANSFER_DST, wait for it to become SHADER_READ_ONLY
			// (do a short polling wait while pumping pending command buffers).
			int waitCount = 0;
			while (textureArrayManager->getLayerLayout(0, targetLayer) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && waitCount < 200) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				if (app) app->processPendingCommandBuffers();
				++waitCount;
			}
			if (waitCount >= 200) {
				std::lock_guard<std::mutex> lkll(logsMutex);
				logs.emplace_back("Warning: waited for transfer dst layout but it persisted");
			}

			// Build barriers for the specific target layer only
			if (genA) {
				auto b = mkBarrierLayer(textureArrayManager->albedoArray.image, targetLayer, 1, textureArrayManager->albedoArray.mipLevels);
				VkImageLayout tracked = textureArrayManager->getLayerLayout(0, targetLayer);
				// Avoid using transient TRANSFER_DST layout as barrier oldLayout for compute transitions;
				// prefer SHADER_READ_ONLY as the common prior state to prevent validation mismatches.
				if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				b.oldLayout = tracked;
				barriers.push_back(b);
			}
			if (genN) {
				auto b = mkBarrierLayer(textureArrayManager->normalArray.image, targetLayer, 1, textureArrayManager->normalArray.mipLevels);
				VkImageLayout tracked = textureArrayManager->getLayerLayout(1, targetLayer);
				if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				b.oldLayout = tracked;
				barriers.push_back(b);
			}
			if (genB) {
				auto b = mkBarrierLayer(textureArrayManager->bumpArray.image, targetLayer, 1, textureArrayManager->bumpArray.mipLevels);
				VkImageLayout tracked = textureArrayManager->getLayerLayout(2, targetLayer);
				if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				b.oldLayout = tracked;
				barriers.push_back(b);
			}
		}
	}
	// Set access masks and newLayout, but keep oldLayout from tracked layout
	for (size_t i = 0; i < barriers.size(); ++i) {
		auto &barrier = barriers[i];
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		// Tracked layouts will be updated when the generation fence signals
	}

	if (!useArrayLayer) {
		throw std::runtime_error("TextureMixer: invalid target layer or missing TextureArrayManager");
	}

	// Log precise barrier info for debugging (image -> base layer)
	{
		std::lock_guard<std::mutex> lkll(logsMutex);
		for (auto &b : barriers) {
			char buf[256];
			snprintf(buf, sizeof(buf), "Pre-barrier: image=%p baseLayer=%u layers=%u newLayout=%d",
				(void*)b.image, b.subresourceRange.baseArrayLayer, b.subresourceRange.layerCount, b.newLayout);
			logs.emplace_back(buf);
		}
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		static_cast<uint32_t>(barriers.size()), barriers.data()
	);

		// Already updated tracked layouts above

	printf("[TextureMixer] vkCmdBindPipeline: computePipeline=%p\n", (void*)computePipeline);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descSet, 0, nullptr);

	vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PerlinPushConstants), &pushConstants);

	uint32_t groupCountX = (width + 15) / 16;
	uint32_t groupCountY = (height + 15) / 16;

	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

		if (useArrayLayer && textureArrayManager) {
		// After compute, transition generated images back to SHADER_READ_ONLY_OPTIMAL
		std::vector<VkImageMemoryBarrier> postBarriers;
		// Build explicit post barriers per-map so we can clamp tracked oldLayout values safely
		if (genA) {
			auto b = mkBarrierLayer(textureArrayManager->albedoArray.image, targetLayer, 1, textureArrayManager->albedoArray.mipLevels);
			VkImageLayout tracked = textureArrayManager->getLayerLayout(0, targetLayer);
			if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.oldLayout = tracked;
			b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
			b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			postBarriers.push_back(b);
		}
		if (genN) {
			auto b = mkBarrierLayer(textureArrayManager->normalArray.image, targetLayer, 1, textureArrayManager->normalArray.mipLevels);
			VkImageLayout tracked = textureArrayManager->getLayerLayout(1, targetLayer);
			if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.oldLayout = tracked;
			b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
			b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			postBarriers.push_back(b);
		}
		if (genB) {
			auto b = mkBarrierLayer(textureArrayManager->bumpArray.image, targetLayer, 1, textureArrayManager->bumpArray.mipLevels);
			VkImageLayout tracked = textureArrayManager->getLayerLayout(2, targetLayer);
			if (tracked == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) tracked = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.oldLayout = tracked;
			b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
			b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			postBarriers.push_back(b);
		}

		// log post-barrier info
		{
			std::lock_guard<std::mutex> lkll(logsMutex);
			for (auto &b : postBarriers) {
				char buf[256];
				snprintf(buf, sizeof(buf), "Post-barrier: image=%p baseLayer=%u layers=%u newLayout=%d",
					(void*)b.image, b.subresourceRange.baseArrayLayer, b.subresourceRange.layerCount, b.newLayout);
				logs.emplace_back(buf);
			}
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			static_cast<uint32_t>(postBarriers.size()), postBarriers.data()
		);

		// Prepare target layer's base level for mipmap generation (only for generated maps)
		targetLayer = static_cast<uint32_t>(params.targetLayer);
		if (targetLayer < textureArrayManager->layerAmount) {
			// set pre-conditions: ensure we don't start copying into an array layer that is currently
			// being generated by TextureMixer itself (avoid layout races). If another generation is
			// pending for this layer, wait for it here.
			TextureMixer* g = TextureMixer::getGlobalInstance();
			if (g) {
				if (g->isLayerGenerationPending(targetLayer)) {
					std::lock_guard<std::mutex> lkll(logsMutex);
					char buf[128];
					snprintf(buf, sizeof(buf), "Waiting for generation to finish for layer=%u before mips/copy", targetLayer);
					logs.emplace_back(buf);
					// Block until generation completes
					g->waitForLayerGeneration(targetLayer);
				}
			}
// Sanity checks: ensure textureArrayManager and image handles exist before recording mips
		if (textureArrayManager) {
			if (genA && textureArrayManager->albedoArray.mipLevels > 1 && textureArrayManager->albedoArray.image != VK_NULL_HANDLE) {
				app->recordGenerateMipmaps(commandBuffer, textureArrayManager->albedoArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->albedoArray.mipLevels, 1, targetLayer);
			} else if (genA) {
				std::lock_guard<std::mutex> lkll(logsMutex); logs.emplace_back("Skipping albedo mipgen: missing image or mipLevels <= 1");
			}
			if (genN && textureArrayManager->normalArray.mipLevels > 1 && textureArrayManager->normalArray.image != VK_NULL_HANDLE) {
				app->recordGenerateMipmaps(commandBuffer, textureArrayManager->normalArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->normalArray.mipLevels, 1, targetLayer);
			} else if (genN) {
				std::lock_guard<std::mutex> lkll(logsMutex); logs.emplace_back("Skipping normal mipgen: missing image or mipLevels <= 1");
			}
			if (genB && textureArrayManager->bumpArray.mipLevels > 1 && textureArrayManager->bumpArray.image != VK_NULL_HANDLE) {
				app->recordGenerateMipmaps(commandBuffer, textureArrayManager->bumpArray.image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), textureArrayManager->bumpArray.mipLevels, 1, targetLayer);
			} else if (genB) {
				std::lock_guard<std::mutex> lkll(logsMutex); logs.emplace_back("Skipping bump mipgen: missing image or mipLevels <= 1");
			}
		} else {
			std::lock_guard<std::mutex> lkll(logsMutex); logs.emplace_back("Skipping mipgen: no TextureArrayManager attached");
			}

			// Submit the recorded command buffer asynchronously and get a fence so we can track completion
			VkSemaphore signalSem = VK_NULL_HANDLE;
		VkFence fence = VK_NULL_HANDLE;
		try {
			fence = app->submitCommandBufferAsync(commandBuffer, &signalSem);
		} catch (const std::exception &e) {
			std::lock_guard<std::mutex> lkll(logsMutex);
			char buf[256];
			snprintf(buf, sizeof(buf), "Async submit failed for layer=%u maps=%s%s%s reason=%s", targetLayer, genA?"A":"", genN?"N":"", genB?"B":"", e.what());
			logs.emplace_back(buf);
			// Free the command buffer locally to avoid leak
			if (commandBuffer != VK_NULL_HANDLE) app->endSingleTimeCommands(commandBuffer);
			commandBuffer = VK_NULL_HANDLE;
		}

		if (fence != VK_NULL_HANDLE) {
			// Log submission
			{
				std::lock_guard<std::mutex> lkll(logsMutex);
				char buf[128];
				snprintf(buf, sizeof(buf), "Submitted async generation: layer=%u maps=%s%s%s", targetLayer, genA?"A":"", genN?"N":"", genB?"B":"");
				logs.emplace_back(buf);
			}

			// Record pending fence so we can call back when generation completes
			{
				std::lock_guard<std::mutex> lk(pendingFencesMutex);
				pendingFences.push_back(std::make_tuple(fence, targetLayer));
			}

			// don't end the single-time command here (ownership transferred to VulkanApp)
			commandBuffer = VK_NULL_HANDLE;
		} else {
			// submission failed or no fence; ensure command buffer ended if still present
			if (commandBuffer != VK_NULL_HANDLE) {
				app->endSingleTimeCommands(commandBuffer);
				commandBuffer = VK_NULL_HANDLE;
			}
		}
	} else {
		for (auto &barrier : barriers) {
			barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			static_cast<uint32_t>(barriers.size()), barriers.data()
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

		std::lock_guard<std::mutex> lkll(logsMutex);
		char buf[256];
		snprintf(buf, sizeof(buf), "Generated layer=%u maps=%s%s%s", targetLayer, genA?"A":"", genN?"N":"", genB?"B":"");
		logs.emplace_back(buf);
	}

	printf("Perlin noise generation complete!\n");

	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}
}
