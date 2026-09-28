#include "TextureMixer.hpp"

#include "../vulkan/VulkanApp.hpp"
#include "../utils/FileReader.hpp"
#include "../vulkan/PerlinPushConstants.hpp"
#include "../vulkan/TextureArrayManager.hpp"
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

TextureMixer::TextureMixer() {}

// Global instance pointer (set in init)
static TextureMixer* g_texture_mixer_instance = nullptr;
TextureMixer* TextureMixer::getGlobalInstance() { return g_texture_mixer_instance; }

void TextureMixer::init(VulkanApp* app) {
}

void TextureMixer::init(VulkanApp* app, TextureArrayManager* texArrMgr) {
	this->textureArrayManager = texArrMgr;
	this->width = texArrMgr->width;
	this->height = textureArrayManager->height;
	// Register global instance so other systems can query/wait on layer generations
	g_texture_mixer_instance = this;

	// Create compute pipeline and descriptor sets so we can generate textures on demand
	createComputePipeline(app);
	printf("[EditableTextureSet] Compute pipeline created for editable textures\n");

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
		std::cerr << "[TextureMixer] Skipping generateInitialTextures: no texture arrays available" << std::endl;
		return;
	}
	printf("Enqueuing initial textures for generation (Albedo, Normal, Bump)...\n");
	for (auto &param : mixerParams) {
		std::cerr << "[TextureMixer] generateInitialTextures: enqueueing generation for layer=" << param.targetLayer << std::endl;
		try {
			enqueueGenerate(param);
		} catch (const std::exception &e) {
			std::lock_guard<std::mutex> lk(logsMutex);
			char buf[256];
			snprintf(buf, sizeof(buf), "generateInitialTextures: enqueue failed for layer=%zu reason=%s", param.targetLayer, e.what());
			logs.emplace_back(buf);
			std::cerr << "[TextureMixer] generateInitialTextures: enqueue failed for layer=" << param.targetLayer << " reason=" << e.what() << std::endl;
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
void TextureMixer::flushPendingRequests(VulkanApp* app) {
	std::vector<std::pair<MixerParameters,int>> tasks;
	{
		std::lock_guard<std::mutex> lk(pendingRequestsMutex);
		tasks.swap(pendingRequests);
	}
	for (auto &t : tasks) {
		// For lower-latency, submit generation asynchronously and track fences
		try {
			generatePerlinNoise(app, const_cast<MixerParameters&>(t.first), t.second);
		} catch (const std::exception &e) {
			std::lock_guard<std::mutex> lkll(logsMutex);
			char buf[256];
			snprintf(buf, sizeof(buf), "generate Perlin failed: layer=%zu map=%d reason=%s", t.first.targetLayer, t.second, e.what());
			logs.emplace_back(buf);
			std::cerr << "[TextureMixer] generatePerlinNoise failed: layer=" << t.first.targetLayer << " map=" << t.second << " reason=" << e.what() << std::endl;
		}
	}
}

void TextureMixer::pollPendingGenerations(VulkanApp* app) {
	// Pull any completed fences and promote their logs (check fences BEFORE letting VulkanApp destroy them)
	completed.clear();
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
				std::lock_guard<std::mutex> lkll(logsMutex);
				char buf[256];
				snprintf(buf, sizeof(buf), "CRITICAL: Device lost detected for fence=%p layer=%u. Aborting resource destruction/reuse for this layer!", (void*)f, layer);
				logs.emplace_back(buf);
				std::cerr << "[TextureMixer] " << buf << std::endl;
				// Do not erase the fence here, keep it for diagnostics
				++it;
				continue;
			} else {
				// Defensive: if the fence is not signaled, do NOT destroy or reuse any resource for this layer
				char buf[256];
				snprintf(buf, sizeof(buf), "WARNING: Fence not signaled for fence=%p layer=%u. Resource destruction/reuse is blocked until signaled.", (void*)f, layer);
				std::lock_guard<std::mutex> lkll(logsMutex);
				logs.emplace_back(buf);
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
			textureArrayManager->setLayerLayout(3, layer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			textureArrayManager->setLayerLayout(4, layer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
		size_t reqs = pendingRequests.size();
		size_t fences = pendingFences.size();
		if (reqs != lastLoggedRequests || fences != lastLoggedFences) {
			std::lock_guard<std::mutex> lkll(logsMutex);
			char buf[128];
			snprintf(buf, sizeof(buf), "Pending: requests=%zu fences=%zu", reqs, fences);
			logs.emplace_back(buf);
			lastLoggedRequests = reqs;
			lastLoggedFences = fences;
		}
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

bool TextureMixer::waitForLayerGeneration(VulkanApp* app, uint32_t layer, uint64_t timeoutNs) {
	TextureMixer* g = getGlobalInstance();
	if (!g || !app) return false;
	std::vector<VkFence> fences;
	{
		std::lock_guard<std::mutex> lk(pendingFencesMutex);
		for (auto &t : pendingFences) {
			if (std::get<1>(t) == layer) fences.push_back(std::get<0>(t));
		}
	}
	if (fences.empty()) return false;
	// Poll each fence via VulkanApp::waitFence instead of vkWaitForFences: the
	// validation layer's vkWaitForFences performs an internal state-tracking wait
	// with a finite timeout that spuriously reports INTERNAL-ERROR-VkFence-state-timeout
	// (most likely a validation bug) and aborts the app. Polling vkGetFenceStatus
	// is spec-valid and validation-clean.
	VkResult r = VK_SUCCESS;
	for (VkFence f : fences) {
		VkResult fr = VulkanApp::waitFence(app->getDevice(), f, timeoutNs);
		if (fr != VK_SUCCESS) r = fr;
	}
	if (r == VK_SUCCESS) {
		// process pending generations so state advances
		if (app) app->processPendingCommandBuffers();
		g->pollPendingGenerations(app);
		return true;
	}
	return false;
}

void TextureMixer::cleanup() {
	// Clear global instance pointer
	if (g_texture_mixer_instance == this) g_texture_mixer_instance = nullptr;

	// No editable textures to cleanup when using global texture arrays

	// Clear local handles; VulkanResourceManager will destroy tracked objects
	computePipeline = VK_NULL_HANDLE;
	computePipelineLayout = VK_NULL_HANDLE;
	computeDescriptorSetLayout = VK_NULL_HANDLE;
	computeDescriptorPool = VK_NULL_HANDLE;

	generationDescSet = VK_NULL_HANDLE;

	// clear log buffer
	{
		std::lock_guard<std::mutex> lk(logsMutex);
		logs.clear();
	}
}


void TextureMixer::createComputePipeline(VulkanApp* app) {
	// Perf report 23 C2: the generation dispatch binds SINGLE-LAYER views for
	// the target (storage) and the primary/secondary sources, so the declared
	// descriptor layouts scope to those layers only. The old array-view
	// bindings forced a whole-array GENERAL <-> SHADER_READ sweep per
	// generation, and the persistent triple/per-map/per-layer sets they fed
	// were unreachable; both are gone.
	//
	// Bindings (fixed layout):
	//   0, 4, 5, 8, 9 : storage images - target layer (albedo/normal/bump/roughness/ao)
	//   1, 2, 3, 6, 7 : samplers       - primary layer (same five maps)
	//   10 .. 14      : samplers       - secondary layer (same five maps)
	VkDescriptorSetLayoutBinding bindings[15] = {};
	auto addBinding = [&](uint32_t binding, VkDescriptorType type) {
		bindings[binding].binding = binding;
		bindings[binding].descriptorType = type;
		bindings[binding].descriptorCount = 1;
		bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	};
	for (uint32_t b : {0u, 4u, 5u, 8u, 9u}) addBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	for (uint32_t b : {1u, 2u, 3u, 6u, 7u, 10u, 11u, 12u, 13u, 14u}) addBinding(b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	VkDescriptorBindingFlags bindingFlags[15] = {};
	for (int i = 0; i < 15; ++i) bindingFlags[i] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

	VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{};
	flagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
	flagsCreateInfo.bindingCount = 15;
	flagsCreateInfo.pBindingFlags = bindingFlags;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 15;
	layoutInfo.pBindings = bindings;
	layoutInfo.pNext = &flagsCreateInfo;
	layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

	if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor set layout!");
	}
	app->resources.addDescriptorSetLayout(computeDescriptorSetLayout, "TextureMixer: computeDescriptorSetLayout");

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
	app->resources.addPipelineLayout(computePipelineLayout, "TextureMixer: computePipelineLayout");

	VkShaderModule computeShaderModule = app->getOrCreateShaderModule("shaders/perlin_noise.comp.spv");

	VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = computeShaderModule;
	computeShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = computeShaderStageInfo;
	pipelineInfo.layout = computePipelineLayout;

	if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(), 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline!");
	}
	app->resources.addPipeline(computePipeline, "TextureMixer: computePipeline");

	// Clear local shader module reference; destruction handled by VulkanResourceManager
	computeShaderModule = VK_NULL_HANDLE;

	// One descriptor set is rewritten per generation. Generation is a
	// synchronous submit, so nothing is in flight while it is updated; the
	// update-after-bind flags keep the write legal regardless.
	VkDescriptorPoolSize poolSizes[2] = {};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[0].descriptorCount = 5;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = 10;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	poolInfo.maxSets = 1;
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

	if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor pool!");
	}
	app->resources.addDescriptorPool(computeDescriptorPool, "TextureMixer: computeDescriptorPool");

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = computeDescriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &computeDescriptorSetLayout;
	if (app->allocateDescriptorSetsThreadSafe(&allocInfo, &generationDescSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate mixer generation descriptor set!");
	}
	app->resources.addDescriptorSet(generationDescSet, "TextureMixer: generationDescSet");
}

// Ensure and return the per-layer 2D view for (layer, map). getImTexture
// creates the view/ImGui pair on demand; the view vectors are the
// authoritative handle for compute bindings.
VkImageView TextureMixer::layerViewFor(uint32_t layer, int map) {
	if (!textureArrayManager || layer >= textureArrayManager->layerAmount) return VK_NULL_HANDLE;
	textureArrayManager->getImTexture(layer, map);
	switch (map) {
		case 0: return layer < textureArrayManager->albedoLayerViews.size() ? textureArrayManager->albedoLayerViews[layer] : VK_NULL_HANDLE;
		case 1: return layer < textureArrayManager->normalLayerViews.size() ? textureArrayManager->normalLayerViews[layer] : VK_NULL_HANDLE;
		case 2: return layer < textureArrayManager->bumpLayerViews.size() ? textureArrayManager->bumpLayerViews[layer] : VK_NULL_HANDLE;
		case 3: return layer < textureArrayManager->roughnessLayerViews.size() ? textureArrayManager->roughnessLayerViews[layer] : VK_NULL_HANDLE;
		case 4: return layer < textureArrayManager->aoLayerViews.size() ? textureArrayManager->aoLayerViews[layer] : VK_NULL_HANDLE;
		default: return VK_NULL_HANDLE;
	}
}

void TextureMixer::attachTextureArrayManager(TextureArrayManager* tam) {
	this->textureArrayManager = tam;
	std::cerr << "[TextureMixer] attachTextureArrayManager called: tam=" << (void*)tam << std::endl;
	// With the C2 per-generation bindings there are no persistent sets to
	// refresh: the next generatePerlinNoise call reads the current views.
}

VkDescriptorSet TextureMixer::getPreviewDescriptor(int map) {
	if (!textureArrayManager || editableLayer == UINT32_MAX) {
		throw std::runtime_error("TextureMixer::getPreviewDescriptor requires TextureArrayManager and valid editableLayer");
	}
	ImTextureID id = textureArrayManager->getImTexture(editableLayer, map);
	return (VkDescriptorSet)id;
}

VkDescriptorSet TextureMixer::getPreviewDescriptor(int map, uint32_t layer) {
	if (!textureArrayManager) {
		throw std::runtime_error("TextureMixer::getPreviewDescriptor(layer) requires TextureArrayManager");
	}
	if (layer >= textureArrayManager->layerAmount) {
		throw std::out_of_range("TextureMixer::getPreviewDescriptor: layer out of range");
	}
	ImTextureID id = textureArrayManager->getImTexture(layer, map);
	return (VkDescriptorSet)id;
}

VkDescriptorSet TextureMixer::getNoiseDescriptor(uint32_t layer) {
	// noise is stored in the alpha channel of the generated texture; the mask is
	// identical for albedo/normal/bump so we can just sample the albedo map's
	// alpha component.  TextureArrayManager provides a helper that creates an
	// alpha-swizzled image view suitable for this purpose.
	if (!textureArrayManager) {
		throw std::runtime_error("TextureMixer::getNoiseDescriptor requires TextureArrayManager");
	}
	if (layer >= textureArrayManager->layerAmount) {
		throw std::out_of_range("TextureMixer::getNoiseDescriptor: layer out of range");
	}
	ImTextureID id = textureArrayManager->getImTextureAlpha(layer, 0);
	return (VkDescriptorSet)id;
}

void TextureMixer::generatePerlinNoise(VulkanApp* app, MixerParameters &params, int map) {
	// log immediate sync generation requests too for diagnostics
	{
		std::lock_guard<std::mutex> lkll(logsMutex);
		char buf[192];
		snprintf(buf, sizeof(buf), "Immediate generate called: layer=%zu primary=%u secondary=%u map=%d",
				 params.targetLayer, params.primaryTextureIdx, params.secondaryTextureIdx, map);
		logs.emplace_back(buf);
	}
	if (!app) throw std::runtime_error("TextureMixer::generatePerlinNoise: app is null");
	if (!textureArrayManager) throw std::runtime_error("TextureMixer requires a TextureArrayManager for array-based generation");
	if (generationDescSet == VK_NULL_HANDLE || computePipeline == VK_NULL_HANDLE) {
		throw std::runtime_error("TextureMixer requires the compute pipeline and generation descriptor set");
	}

	const uint32_t targetLayer = static_cast<uint32_t>(params.targetLayer);
	const uint32_t primaryLayer = static_cast<uint32_t>(params.primaryTextureIdx);
	const uint32_t secondaryLayer = static_cast<uint32_t>(params.secondaryTextureIdx);
	if (targetLayer == primaryLayer || targetLayer == secondaryLayer) {
		return; // avoid sampling and writing to the same layer
	}
	if (targetLayer >= textureArrayManager->layerAmount ||
		primaryLayer >= textureArrayManager->layerAmount ||
		secondaryLayer >= textureArrayManager->layerAmount) {
		throw std::runtime_error("TextureMixer::generatePerlinNoise: layer index out of range");
	}

	// Single-layer views for the three layers this generation touches. The
	// map argument is retained for interface compatibility: the shader writes
	// all five maps of the target layer, so the full write set is bound and
	// regenerated (the old per-map filter transitioned one map while the
	// shader wrote five, which was a latent layout violation).
	VkImageView targetViews[5] = {};
	VkImageView primaryViews[5] = {};
	VkImageView secondaryViews[5] = {};
	for (int m = 0; m < 5; ++m) {
		targetViews[m] = layerViewFor(targetLayer, m);
		primaryViews[m] = layerViewFor(primaryLayer, m);
		secondaryViews[m] = layerViewFor(secondaryLayer, m);
		if (targetViews[m] == VK_NULL_HANDLE || primaryViews[m] == VK_NULL_HANDLE || secondaryViews[m] == VK_NULL_HANDLE) {
			throw std::runtime_error("TextureMixer::generatePerlinNoise: per-layer views unavailable");
		}
	}

	// Re-point the generation set at this request's layers. Storage views
	// declare GENERAL (the target layer is transitioned before the dispatch);
	// the sampled primary/secondary layers stay SHADER_READ_ONLY and are
	// never transitioned.
	VkSampler mapSamplers[5] = {
		textureArrayManager->albedoSampler, textureArrayManager->normalSampler,
		textureArrayManager->bumpSampler, textureArrayManager->roughnessSampler,
		textureArrayManager->aoSampler
	};
	const uint32_t storageBindings[5] = { 0, 4, 5, 8, 9 };
	const uint32_t primaryBindings[5] = { 1, 2, 3, 6, 7 };
	const uint32_t secondaryBindings[5] = { 10, 11, 12, 13, 14 };
	VkDescriptorImageInfo storageInfos[5] = {};
	VkDescriptorImageInfo primaryInfos[5] = {};
	VkDescriptorImageInfo secondaryInfos[5] = {};
	for (int m = 0; m < 5; ++m) {
		storageInfos[m].imageView = targetViews[m];
		storageInfos[m].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		primaryInfos[m].imageView = primaryViews[m];
		primaryInfos[m].sampler = mapSamplers[m];
		primaryInfos[m].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		secondaryInfos[m].imageView = secondaryViews[m];
		secondaryInfos[m].sampler = mapSamplers[m];
		secondaryInfos[m].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	std::vector<VkWriteDescriptorSet> writes;
	writes.reserve(15);
	for (int m = 0; m < 5; ++m) {
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = generationDescSet;
		w.descriptorCount = 1;
		w.dstBinding = storageBindings[m];
		w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		w.pImageInfo = &storageInfos[m];
		writes.push_back(w);
		w.dstBinding = primaryBindings[m];
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.pImageInfo = &primaryInfos[m];
		writes.push_back(w);
		w.dstBinding = secondaryBindings[m];
		w.pImageInfo = &secondaryInfos[m];
		writes.push_back(w);
	}
	vkUpdateDescriptorSets(app->getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	// Dimensions are read live: a resolution-tier rebuild (perf report 23 C1)
	// swaps the arrays underneath the mixer.
	const uint32_t texW = textureArrayManager->width;
	const uint32_t texH = textureArrayManager->height;

	PerlinPushConstants pushConstants{};
	pushConstants.scale = params.perlinScale;
	pushConstants.octaves = params.perlinOctaves;
	pushConstants.persistence = params.perlinPersistence;
	pushConstants.lacunarity = params.perlinLacunarity;
	pushConstants.brightness = params.perlinBrightness;
	pushConstants.contrast = params.perlinContrast;
	pushConstants.seed = params.perlinSeed;
	pushConstants.textureSize = texW;
	pushConstants.time = params.perlinTime;

	// Generation is synchronous (runSingleTimeCommands blocks on the fence),
	// so no previous dispatch can still reference the set we just updated;
	// pumping completed work keeps the deferred layout bookkeeping current.
	app->processPendingCommandBuffers();

	const VkImage images[5] = {
		textureArrayManager->albedoArray.image, textureArrayManager->normalArray.image,
		textureArrayManager->bumpArray.image, textureArrayManager->roughnessArray.image,
		textureArrayManager->aoArray.image
	};
	const uint32_t mips[5] = {
		textureArrayManager->albedoArray.mipLevels, textureArrayManager->normalArray.mipLevels,
		textureArrayManager->bumpArray.mipLevels, textureArrayManager->roughnessArray.mipLevels,
		textureArrayManager->aoArray.mipLevels
	};

	app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
		// Target layer -> GENERAL only (C2): no non-target sweep. The
		// primary/secondary layers are sampled from their SHADER_READ_ONLY
		// single-layer views and never move.
		for (int m = 0; m < 5; ++m) {
			app->recordTransitionImageLayoutLayer(cmd, images[m], VK_FORMAT_R8G8B8A8_UNORM,
				textureArrayManager->getLayerLayout(m, targetLayer), VK_IMAGE_LAYOUT_GENERAL,
				mips[m], targetLayer, 1);
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &generationDescSet, 0, nullptr);
		vkCmdPushConstants(cmd, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PerlinPushConstants), &pushConstants);

		const uint32_t groupCountX = (texW + 15) / 16;
		const uint32_t groupCountY = (texH + 15) / 16;
		vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

		// Prepare the written layer for mipmap generation and record the five
		// chains into the same command buffer.
		for (int m = 0; m < 5; ++m) {
			app->recordTransitionImageLayoutLayer(cmd, images[m], VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				mips[m], targetLayer, 1);
		}
		for (int m = 0; m < 5; ++m) {
			if (mips[m] > 1) {
				app->recordGenerateMipmaps(cmd, images[m], VK_FORMAT_R8G8B8A8_UNORM,
					static_cast<int32_t>(texW), static_cast<int32_t>(texH), mips[m], 1, targetLayer);
			} else {
				app->recordTransitionImageLayoutLayer(cmd, images[m], VK_FORMAT_R8G8B8A8_UNORM,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					1, targetLayer, 1);
			}
		}
	});

	// Synchronous generation complete: mark the layer initialized and track
	// the final layouts.
	textureArrayManager->setLayerInitialized(targetLayer, true);
	for (int m = 0; m < 5; ++m) {
		textureArrayManager->setLayerLayout(m, targetLayer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	if (onTextureGeneratedCallback) {
		onTextureGeneratedCallback();
	}
}
