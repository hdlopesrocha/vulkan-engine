// Minimal implementation file for TextureArrayManager
#include "TextureArrayManager.hpp"

#include "../core/VulkanApp.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>
#include <cmath>
#include <stb/stb_image.h>
// <tuple> no longer required (using TextureTriple)

// Convert in-place 8-bit RGBA sRGB values to linear (also 8-bit)
void convertSRGB8ToLinearInPlace(unsigned char* data, size_t pixelCount) {
    for (size_t i = 0; i < pixelCount; ++i) {
        unsigned char* p = data + i * 4;
        for (int c = 0; c < 3; ++c) {
            float srgb = p[c] / 255.0f;
            float lin = (srgb <= 0.04045f) ? (srgb / 12.92f) : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
            int v = static_cast<int>(std::round(lin * 255.0f));
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            p[c] = static_cast<unsigned char>(v);
        }
        // alpha channel left as-is
    }
}


void TextureArrayManager::notifyAllocationListeners() {
	std::vector<std::function<void()>> listenersCopy;
	listenersCopy.reserve(allocationListeners.size());
	for (auto &l : allocationListeners) if (l) listenersCopy.push_back(l);
	for (size_t i = 0; i < listenersCopy.size(); ++i) {
		try {
			listenersCopy[i]();
		} catch (const std::exception &e) {
			std::cerr << "[TextureArrayManager] allocation listener " << i << " threw std::exception: " << e.what() << std::endl;
		} catch (...) {
			std::cerr << "[TextureArrayManager] allocation listener " << i << " threw unknown exception" << std::endl;
		}
	}
}

// Helper to cleanup a TextureImage if it already has resources.
// Always defers GPU resource destruction via deferDestroyUntilAllPending
// to ensure in-flight graphics frames (tracked by the timeline semaphore)
// have completed before freeing memory.  The old check using
// hasPendingCommandBuffers() only considered async submissions, not
// in-flight graphics frames, and could destroy memory still accessible
// by the GPU — causing GPUVM faults and VK_ERROR_DEVICE_LOST.
static void cleanupTextureImage(VulkanApp* app, TextureImage &ti) {
	if (!app) return;
	VkDevice device = app->getDevice();
	// Destroy view
	if (ti.view != VK_NULL_HANDLE) {
		VkImageView v = ti.view;
		app->deferDestroyUntilAllPending([device, v, app](){ if (app->resources.removeImageView(v)) vkDestroyImageView(device, v, nullptr); });
		ti.view = VK_NULL_HANDLE;
	}
	// Destroy image + memory via VMA if allocated through VMA
	if (ti.image != VK_NULL_HANDLE) {
		VkImage img = ti.image;
		VmaAllocation alloc = ti.allocation;
		VkDeviceMemory mem = ti.memory;
		app->deferDestroyUntilAllPending([device, img, alloc, mem, app](){ app->destroyImageWithVma(img, alloc, mem); });
		ti.image = VK_NULL_HANDLE;
		ti.allocation = VK_NULL_HANDLE;
		ti.memory = VK_NULL_HANDLE;
	}
	ti.mipLevels = 1;
}

// Cleanup sampler if present.
// Always defers GPU resource destruction via deferDestroyUntilAllPending
// to ensure in-flight graphics frames have completed before freeing.
static void cleanupSampler(VulkanApp* app, VkSampler &s) {
	if (!app) return;
	VkDevice device = app->getDevice();
	if (s != VK_NULL_HANDLE) {
		VkSampler ss = s;
		app->deferDestroyUntilAllPending([device, ss, app](){ if (app->resources.removeSampler(ss)) vkDestroySampler(device, ss, nullptr); });
		s = VK_NULL_HANDLE;
	}
}

// Release the per-layer 2D views and their ImGui descriptors. Always defers
// destruction to avoid freeing resources the current frame (or any in-flight
// frame) may still reference. Split out of destroy() so recreate() can drop
// the views that belong to the old array images before allocate() replaces
// them (perf report 23 C1). Does NOT notify allocation listeners.
void TextureArrayManager::releaseLayerViews(VulkanApp* app) {
	if (!app) return;
	VkDevice device = app->getDevice();
	// Remove any ImGui textures and destroy per-layer views.
	// Always defer removal to avoid destroying descriptor sets that may still
	// be referenced by the current frame's (or any in-flight) command buffer.
	for (auto &tex : albedoImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
			tex = 0;
		}
	}
	for (auto &tex : normalImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
			tex = 0;
		}
	}
	for (auto &tex : bumpImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
			tex = 0;
		}
	}
	for (auto &tex : roughnessImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
			tex = 0;
		}
	}
	for (auto &tex : aoImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
			tex = 0;
		}
	}
	// Destroy per-layer views; always defer to avoid destroying resources that
	// the current frame may still reference.
	for (auto &v : albedoLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			app->deferDestroyUntilAllPending([device, iv, app](){ if (app->resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr); });
			v = VK_NULL_HANDLE;
		}
	}
	for (auto &v : normalLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			app->deferDestroyUntilAllPending([device, iv, app](){ if (app->resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr); });
			v = VK_NULL_HANDLE;
		}
	}
	for (auto &v : bumpLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			app->deferDestroyUntilAllPending([device, iv, app](){ if (app->resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr); });
			v = VK_NULL_HANDLE;
		}
	}
	for (auto &v : roughnessLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			app->deferDestroyUntilAllPending([device, iv, app](){ if (app->resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr); });
			v = VK_NULL_HANDLE;
		}
	}
	for (auto &v : aoLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			app->deferDestroyUntilAllPending([device, iv, app](){ if (app->resources.removeImageView(iv)) vkDestroyImageView(device, iv, nullptr); });
			v = VK_NULL_HANDLE;
		}
	}
	albedoLayerViews.clear(); normalLayerViews.clear(); bumpLayerViews.clear(); roughnessLayerViews.clear(); aoLayerViews.clear();
	albedoImTextures.clear(); normalImTextures.clear(); bumpImTextures.clear(); roughnessImTextures.clear(); aoImTextures.clear();
}

void TextureArrayManager::destroy(VulkanApp* app) {
	if (!app) return;
	cleanupTextureImage(app, albedoArray);
	cleanupTextureImage(app, normalArray);
	cleanupTextureImage(app, bumpArray);
	cleanupTextureImage(app, roughnessArray);
	cleanupTextureImage(app, aoArray);
	cleanupSampler(app, albedoSampler);
	cleanupSampler(app, normalSampler);
	cleanupSampler(app, bumpSampler);
	cleanupSampler(app, roughnessSampler);
	cleanupSampler(app, aoSampler);
	releaseStagingBuffer(app);
	releaseLayerViews(app);
	// clear stored app pointer (no longer valid after destroy)
	this->appPtr = nullptr;
	// bump version to indicate array resources were destroyed
	++this->version;
	// notify listeners that arrays were destroyed
	notifyAllocationListeners();
}

void TextureArrayManager::recreate(VulkanApp* app, uint32_t layers, uint32_t w, uint32_t h) {
	if (!app) throw std::runtime_error("TextureArrayManager::recreate: app is null");
	// Drop the views that reference the old images first (deferred), then let
	// allocate() replace the images and notify listeners exactly once with the
	// new views. The caller must hold a device idle and drain deferred
	// destroys afterwards; no listener observes the transient null-view state
	// because only allocate() notifies.
	releaseLayerViews(app);
	allocate(layers, w, h, app);
}

void TextureArrayManager::logMemoryUtilization(const char* name) const {
	if (layerAmount == 0) return;
	const char* label = name ? name : "arrays";
	// Bytes of one layer of an RGBA8 mip chain. The five material arrays are
	// all R8G8B8A8_UNORM; compute per image so a future format change cannot
	// silently lie in the log.
	auto layerBytes = [&](const TextureImage& img) -> size_t {
		uint32_t w = std::max(1u, width);
		uint32_t h = std::max(1u, height);
		const uint32_t mips = std::max(1u, img.mipLevels);
		size_t total = 0;
		for (uint32_t m = 0; m < mips; ++m) {
			total += static_cast<size_t>(w) * h * 4u;
			w = std::max(1u, w / 2);
			h = std::max(1u, h / 2);
		}
		return total;
	};
	const TextureImage* imgs[5] = { &albedoArray, &normalArray, &bumpArray, &roughnessArray, &aoArray };
	const char* mapNames[5] = { "albedo", "normal", "bump", "roughness", "ao" };
	size_t perArray[5] = { 0, 0, 0, 0, 0 };
	size_t allocated = 0;
	for (int i = 0; i < 5; ++i) {
		perArray[i] = layerBytes(*imgs[i]) * layerAmount;
		allocated += perArray[i];
	}
	size_t usedLayers = 0;
	for (char c : layerInitialized) if (c) ++usedLayers;
	if (usedLayers > layerAmount) usedLayers = layerAmount;
	const size_t used = layerAmount ? (allocated * usedLayers) / layerAmount : 0;
	const double toMB = 1.0 / (1024.0 * 1024.0);
	std::printf("[memutil] texture arrays '%s': %ux%u x %u layers x 5 maps: %.0f / %.0f MB used (%.0f%% layers)\n",
	            label, width, height, layerAmount,
	            static_cast<double>(used) * toMB, static_cast<double>(allocated) * toMB,
	            layerAmount ? 100.0 * static_cast<double>(usedLayers) / static_cast<double>(layerAmount) : 0.0);
	std::printf("[memutil]   %s %.0f, %s %.0f, %s %.0f, %s %.0f, %s %.0f MB allocated\n",
	            mapNames[0], static_cast<double>(perArray[0]) * toMB,
	            mapNames[1], static_cast<double>(perArray[1]) * toMB,
	            mapNames[2], static_cast<double>(perArray[2]) * toMB,
	            mapNames[3], static_cast<double>(perArray[3]) * toMB,
	            mapNames[4], static_cast<double>(perArray[4]) * toMB);
	if (usedLayers > 0 && usedLayers * 4 < layerAmount) {
		std::printf("[memutil] texture arrays '%s': WARNING <25%% of allocated layers are initialized (%zu/%u)\n",
		            label, usedLayers, layerAmount);
	}
}

void TextureArrayManager::allocate(uint32_t layers, uint32_t w, uint32_t h, VulkanApp* app) {
	if (!app) throw std::runtime_error("TextureArrayManager::allocate: app is null");

	// Store back-pointer for legacy UI convenience (ImGui descriptors)
	this->appPtr = app;

	layerAmount = layers;
	width = w;
	height = h;
	// A fresh allocation starts empty. Without this reset a recreate() would
	// re-load triples at the stale cursor and stop at the old capacity,
	// leaving the array partially filled (perf report 23 C1 follow-up).
	currentLayer = 0;
	releaseStagingBuffer(app);

	// destroy previous resources if present
	cleanupTextureImage(app, albedoArray);
	cleanupTextureImage(app, normalArray);
	cleanupTextureImage(app, bumpArray);
	cleanupTextureImage(app, roughnessArray);
	cleanupTextureImage(app, aoArray);

	// Compute mip level count and create 2D array image
	uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

	auto createArray = [&](TextureImage &out, VkFormat format, bool srgb){
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = layerAmount;
		imageInfo.format = format;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

		VkDevice device = app->getDevice();

		app->createImageWithVma(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out.image, out.allocation, out.memory, "TextureArrayManager: createArray");
		app->resources.setImageArrayLayers(out.image, imageInfo.arrayLayers);


		// Create image view for 2D array
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = out.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = layerAmount;

		if (vkCreateImageView(device, &viewInfo, nullptr, &out.view) != VK_SUCCESS) {
			app->destroyImageWithVma(out.image, out.allocation, out.memory);
			out.image = VK_NULL_HANDLE;
			out.allocation = VK_NULL_HANDLE;
			out.memory = VK_NULL_HANDLE;
			throw std::runtime_error("failed to create texture array image view");
		}
		app->resources.addImageView(out.view, "TextureArrayManager::createArray view");

		out.mipLevels = mipLevels;

		// Transition ALL mip levels and array layers to TRANSFER_DST then to SHADER_READ_ONLY
		app->transitionImageLayout(out.image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels, layerAmount);
		app->transitionImageLayout(out.image, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels, layerAmount);
	};

	// Albedo: use UNORM format (no automatic sRGB->linear conversion)
	createArray(albedoArray, VK_FORMAT_R8G8B8A8_UNORM, true);
	// Normal and bump maps use UNORM
	createArray(normalArray, VK_FORMAT_R8G8B8A8_UNORM, false);
	createArray(bumpArray, VK_FORMAT_R8G8B8A8_UNORM, false);
	createArray(roughnessArray, VK_FORMAT_R8G8B8A8_UNORM, false);
	createArray(aoArray, VK_FORMAT_R8G8B8A8_UNORM, false);

	// cleanup existing samplers and create new ones
	cleanupSampler(app, albedoSampler);
	cleanupSampler(app, normalSampler);
	cleanupSampler(app, bumpSampler);
	cleanupSampler(app, roughnessSampler);
	cleanupSampler(app, aoSampler);

	albedoSampler = app->createTextureSampler(mipLevels);
	normalSampler = app->createTextureSampler(mipLevels);
	bumpSampler = app->createTextureSampler(mipLevels);
	roughnessSampler = app->createTextureSampler(mipLevels);
	aoSampler = app->createTextureSampler(mipLevels);
	// Do NOT store `app` in this manager; callers pass `app` explicitly to GPU operations
	(void)app; // keep parameter used, but don't retain pointer
	// initialize layer initialized flags
	layerInitialized.clear();
	layerInitialized.resize(layerAmount, 0);
	// (re)size the per-layer albedo averages (mid-gray until loaded)
	albedoAvg.clear();
	albedoAvg.resize(layerAmount, {0.5f, 0.5f, 0.5f});

	// initialize per-layer layout trackers to UNDEFINED by default
	albedoLayerLayouts.clear(); albedoLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	normalLayerLayouts.clear(); normalLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	bumpLayerLayouts.clear(); bumpLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	roughnessLayerLayouts.clear(); roughnessLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	aoLayerLayouts.clear(); aoLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);

	// After creating arrays we transitioned all mips/layers to SHADER_READ_ONLY_OPTIMAL above; reflect that state
	for (uint32_t i = 0; i < layerAmount; ++i) {
		albedoLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		normalLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		bumpLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		roughnessLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		aoLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	// bump version so users can detect reallocation of GPU resources
	++this->version;
	// notify listeners that arrays changed
	notifyAllocationListeners();
}

// Mean linear RGB of an RGBA8 buffer (values already 0..255 linear).
static std::array<float, 3> meanLinearRGB(const unsigned char* data, size_t pixelCount) {
	std::array<float, 3> avg{0.5f, 0.5f, 0.5f};
	if (!data || pixelCount == 0) return avg;
	double r = 0.0, g = 0.0, b = 0.0;
	for (size_t p = 0; p < pixelCount; ++p) {
		r += data[p * 4 + 0];
		g += data[p * 4 + 1];
		b += data[p * 4 + 2];
	}
	const float inv = 1.0f / (255.0f * static_cast<float>(pixelCount));
	avg[0] = static_cast<float>(r * inv);
	avg[1] = static_cast<float>(g * inv);
	avg[2] = static_cast<float>(b * inv);
	return avg;
}

// Area-average resample (RGBA8). Each destination texel averages the source
// rectangle that maps onto it; an exact 2:1 downscale (1024 -> 512, the tier
// and vegetation sizes) is a clean 2x2 box. Upscales collapse to the nearest
// source texel. Replaces the previous nearest-neighbour path, which
// point-sampled 2:1 — aliasing alpha-cut vegetation and breaking normal/bump
// detail (perf report 23 C1; also closes M11's downscale bullet).
static unsigned char* resizeAreaAverage(const unsigned char* src, int srcW, int srcH, int dstW, int dstH) {
	unsigned char* dst = new unsigned char[static_cast<size_t>(dstW) * dstH * 4];
	const double sx = static_cast<double>(srcW) / dstW;
	const double sy = static_cast<double>(srcH) / dstH;
	for (int y = 0; y < dstH; ++y) {
		int y0 = static_cast<int>(y * sy);
		int y1 = static_cast<int>((y + 1) * sy);
		y0 = std::min(std::max(y0, 0), srcH - 1);
		y1 = std::min(std::max(y1, y0 + 1), srcH);
		for (int x = 0; x < dstW; ++x) {
			int x0 = static_cast<int>(x * sx);
			int x1 = static_cast<int>((x + 1) * sx);
			x0 = std::min(std::max(x0, 0), srcW - 1);
			x1 = std::min(std::max(x1, x0 + 1), srcW);
			uint32_t r = 0, g = 0, b = 0, a = 0, n = 0;
			for (int yy = y0; yy < y1; ++yy) {
				const unsigned char* row = src + (static_cast<size_t>(yy) * srcW + x0) * 4;
				for (int xx = x0; xx < x1; ++xx) {
					r += row[0]; g += row[1]; b += row[2]; a += row[3];
					row += 4;
					++n;
				}
			}
			unsigned char* d = dst + (static_cast<size_t>(y) * dstW + x) * 4;
			d[0] = static_cast<unsigned char>(r / n);
			d[1] = static_cast<unsigned char>(g / n);
			d[2] = static_cast<unsigned char>(b / n);
			d[3] = static_cast<unsigned char>(a / n);
		}
	}
	return dst;
}

// Perf report 23 C3: persistent staging for layer uploads. One host-visible
// buffer holds all five maps of one layer; every load() overwrites it in
// place, so the bring-up path performs zero staging allocations/frees and no
// zero-init memset (the caller writes every byte it uploads).
Buffer& TextureArrayManager::ensureStagingBuffer(VulkanApp* app, size_t needBytes) {
	if (stagingBuffer_.buffer == VK_NULL_HANDLE || stagingBufferSize_ < needBytes) {
		releaseStagingBuffer(app);
		// zeroInit=false: load() overwrites the full range before every copy;
		// the default zero-fill would memset needBytes for nothing (420 MiB
		// across the 21-triple bring-up, report 23 C3).
		stagingBuffer_ = app->createBuffer(needBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
		stagingBufferSize_ = needBytes;
	}
	return stagingBuffer_;
}

void TextureArrayManager::releaseStagingBuffer(VulkanApp* app) {
	if (app && stagingBuffer_.buffer != VK_NULL_HANDLE) {
		app->destroyBuffer(stagingBuffer_);
	}
	stagingBuffer_ = {};
	stagingBufferSize_ = 0;
}

uint TextureArrayManager::load(VulkanApp* a, const char* albedoFile, const char* normalFile, const char* bumpFile, const char* roughnessFile, const char* aoFile) {
	if (!a) throw std::runtime_error("TextureArrayManager::load: app is null");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::load: layerAmount == 0");
	if (currentLayer >= layerAmount) throw std::runtime_error("TextureArrayManager::load: currentLayer >= layerAmount");

	struct Img { const char* path; TextureImage* dstImage; VkFormat format; bool srgb; unsigned char defaultVal[4]; } imgs[5] = {
		{ albedoFile, &albedoArray, VK_FORMAT_R8G8B8A8_UNORM, true, {0,0,0,255} },
		{ normalFile, &normalArray, VK_FORMAT_R8G8B8A8_UNORM, false, {128,128,255,255} },
		{ bumpFile,   &bumpArray,   VK_FORMAT_R8G8B8A8_UNORM, false, {128,128,128,255} },
		{ roughnessFile, &roughnessArray, VK_FORMAT_R8G8B8A8_UNORM, false, {128,128,128,255} },
		{ aoFile,     &aoArray,     VK_FORMAT_R8G8B8A8_UNORM, false, {255,255,255,255} }
	};

	// Decode every present map straight into its slice of the persistent
	// staging buffer (one buffer for all five maps, reused for every layer).
	const size_t layerBytes = static_cast<size_t>(width) * height * 4;
	Buffer& staging = ensureStagingBuffer(a, layerBytes * 5);

	for (int i = 0; i < 5; ++i) {
		unsigned char* dst = static_cast<unsigned char*>(staging.mappedData) + layerBytes * static_cast<size_t>(i);
		if (!imgs[i].path) {
			for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
				memcpy(dst + p * 4, imgs[i].defaultVal, 4);
			}
			continue;
		}
		int texW = 0, texH = 0, texC = 0;
		unsigned char* pixels = stbi_load(imgs[i].path, &texW, &texH, &texC, 4);
		if (!pixels) {
			throw std::runtime_error(std::string("failed to load texture: ") + imgs[i].path);
		}
		if (texW != static_cast<int>(width) || texH != static_cast<int>(height)) {
			unsigned char* resized = resizeAreaAverage(pixels, texW, texH, static_cast<int>(width), static_cast<int>(height));
			memcpy(dst, resized, layerBytes);
			delete[] resized;
		} else {
			memcpy(dst, pixels, layerBytes);
		}
		stbi_image_free(pixels);
		if (imgs[i].srgb) {
			convertSRGB8ToLinearInPlace(dst, static_cast<size_t>(width) * static_cast<size_t>(height));
		}
	}

	// Record the albedo map's average (linear after the conversion above).
	if (currentLayer < albedoAvg.size()) {
		albedoAvg[currentLayer] = meanLinearRGB(
			static_cast<const unsigned char*>(staging.mappedData),
			static_cast<size_t>(width) * static_cast<size_t>(height));
	}

	// C3: one command buffer and one blocking submit per layer -- five copies
	// (all maps) plus their mip chains, instead of one submit per map plus a
	// separate blocking mip submit each.
	a->runSingleTimeCommands([&](VkCommandBuffer cmd) {
		for (int i = 0; i < 5; ++i) {
			a->recordTransitionImageLayoutLayer(cmd, imgs[i].dstImage->image, imgs[i].format,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				imgs[i].dstImage->mipLevels, currentLayer, 1);

			VkBufferImageCopy region{};
			region.bufferOffset = static_cast<VkDeviceSize>(layerBytes) * static_cast<VkDeviceSize>(i);
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = currentLayer;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = {0,0,0};
			region.imageExtent = { width, height, 1 };
			vkCmdCopyBufferToImage(cmd, staging.buffer, imgs[i].dstImage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		}
		for (int i = 0; i < 5; ++i) {
			if (imgs[i].dstImage->mipLevels > 1) {
				a->recordGenerateMipmaps(cmd, imgs[i].dstImage->image, imgs[i].format,
					static_cast<int32_t>(width), static_cast<int32_t>(height),
					imgs[i].dstImage->mipLevels, 1, currentLayer);
			} else {
				a->recordTransitionImageLayoutLayer(cmd, imgs[i].dstImage->image, imgs[i].format,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					1, currentLayer, 1);
			}
		}
	});

	for (int i = 0; i < 5; ++i) {
		setLayerLayout(i, currentLayer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	setLayerInitialized(currentLayer, true);
	return currentLayer++;
}

size_t TextureArrayManager::loadTriples(VulkanApp* a, const std::vector<TextureTriple> &triples) {
	if (!a) throw std::runtime_error("TextureArrayManager::loadTriples: app is null");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::loadTriples: layerAmount == 0");
	size_t loaded = 0;
	for (const auto &t : triples) {
		if (currentLayer >= layerAmount) {
			std::cerr << "[TextureArrayManager] Reached texture array capacity (" << layerAmount << " layers)" << std::endl;
			break;
		}
		try {
			load(a, t.albedo, t.normal, t.bump, t.roughness, t.ao);
			++loaded;
		} catch (const std::exception &e) {
			std::cerr << "[TextureArrayManager] Failed to load " << (t.albedo ? t.albedo : "(null)") << ": " << e.what() << std::endl;
		}
	}
	return loaded;
}

// Per-layer layout helpers
VkImageLayout TextureArrayManager::getLayerLayout(int map, uint32_t layer) const {
	if (layer >= layerAmount) return VK_IMAGE_LAYOUT_UNDEFINED;
	switch (map) {
		case 0: return (albedoLayerLayouts.size() > layer) ? albedoLayerLayouts[layer] : VK_IMAGE_LAYOUT_UNDEFINED;
		case 1: return (normalLayerLayouts.size() > layer) ? normalLayerLayouts[layer] : VK_IMAGE_LAYOUT_UNDEFINED;
		case 2: return (bumpLayerLayouts.size() > layer) ? bumpLayerLayouts[layer] : VK_IMAGE_LAYOUT_UNDEFINED;
		case 3: return (roughnessLayerLayouts.size() > layer) ? roughnessLayerLayouts[layer] : VK_IMAGE_LAYOUT_UNDEFINED;
		case 4: return (aoLayerLayouts.size() > layer) ? aoLayerLayouts[layer] : VK_IMAGE_LAYOUT_UNDEFINED;
		default: return VK_IMAGE_LAYOUT_UNDEFINED;
	}
}

void TextureArrayManager::setLayerLayout(int map, uint32_t layer, VkImageLayout layout) {
	if (layer >= layerAmount) return;
	switch (map) {
		case 0: if (albedoLayerLayouts.size() > layer) albedoLayerLayouts[layer] = layout; break;
		case 1: if (normalLayerLayouts.size() > layer) normalLayerLayouts[layer] = layout; break;
		case 2: if (bumpLayerLayouts.size() > layer) bumpLayerLayouts[layer] = layout; break;
		case 3: if (roughnessLayerLayouts.size() > layer) roughnessLayerLayouts[layer] = layout; break;
		case 4: if (aoLayerLayouts.size() > layer) aoLayerLayouts[layer] = layout; break;
		default: break;
	}
}

int TextureArrayManager::addAllocationListener(std::function<void()> cb) {	// find an empty slot or push
	for (size_t i = 0; i < allocationListeners.size(); ++i) {
		if (!allocationListeners[i]) {
			allocationListeners[i] = cb;
			return static_cast<int>(i);
		}
	}
	allocationListeners.push_back(cb);
	return static_cast<int>(allocationListeners.size() - 1);
}

void TextureArrayManager::removeAllocationListener(int listenerId) {
	if (listenerId < 0) return;
	auto idx = static_cast<size_t>(listenerId);
	if (idx < allocationListeners.size()) allocationListeners[idx] = {};
}

void TextureArrayManager::invalidateImGuiDescriptors() {
    auto clear = [](auto& vec) { std::fill(vec.begin(), vec.end(), ImTextureID(0)); };
    clear(albedoImTextures);
    clear(normalImTextures);
    clear(bumpImTextures);
    clear(roughnessImTextures);
    clear(aoImTextures);
}

bool TextureArrayManager::isLayerInitialized(uint32_t layer) const {
	if (layer >= layerInitialized.size()) return false;
	return layerInitialized[layer] != 0;
}

void TextureArrayManager::setLayerInitialized(uint32_t layer, bool v) {
	if (layer >= layerInitialized.size()) return;
	layerInitialized[layer] = v ? 1 : 0;
}

std::array<float, 3> TextureArrayManager::albedoAverage(uint32_t layer) const {
	static const std::array<float, 3> kFallback{0.5f, 0.5f, 0.5f};
	if (layer >= albedoAvg.size()) return kFallback;
	return albedoAvg[layer];
}

ImTextureID TextureArrayManager::getImTexture(size_t layer, int map) {
	if (layer >= layerAmount) return 0;
	VulkanApp* a = this->appPtr;
	if (!a) return 0;
	VkDevice device = a->getDevice();

	std::vector<VkImageView>* viewVec = nullptr;
	std::vector<ImTextureID>* texVec = nullptr;
	TextureImage* src = nullptr;
	VkSampler sampler = VK_NULL_HANDLE;
	switch (map) {
		case 0: viewVec = &albedoLayerViews; texVec = &albedoImTextures; src = &albedoArray; sampler = albedoSampler; break;
		case 1: viewVec = &normalLayerViews; texVec = &normalImTextures; src = &normalArray; sampler = normalSampler; break;
		case 2: viewVec = &bumpLayerViews; texVec = &bumpImTextures; src = &bumpArray; sampler = bumpSampler; break;
		case 3: viewVec = &roughnessLayerViews; texVec = &roughnessImTextures; src = &roughnessArray; sampler = roughnessSampler; break;
		case 4: viewVec = &aoLayerViews; texVec = &aoImTextures; src = &aoArray; sampler = aoSampler; break;
		default: return 0;
	}

	// ensure vectors are sized
	if (viewVec->size() != layerAmount) viewVec->resize(layerAmount, VK_NULL_HANDLE);
	if (texVec->size() != layerAmount) texVec->resize(layerAmount, 0);

	// if ImTextureID already created, return it
	if ((*texVec)[layer]) return (*texVec)[layer];

	// create a per-layer 2D image view
		if (!(*viewVec)[layer]) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = src->image;
			// Create a simple 2D view for this layer. ImGui shaders sample via
			// sampler2D, so the view must not be arrayed.
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		// choose format consistent with array creation
			viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = src->mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = static_cast<uint32_t>(layer);
		viewInfo.subresourceRange.layerCount = 1;
		// Use the source image's format where possible
		// Attempt to read format from src - not stored publicly here; assume appropriate format
		if (vkCreateImageView(device, &viewInfo, nullptr, &(*viewVec)[layer]) != VK_SUCCESS) {
			return 0;
		}
				// Register per-layer view so centralized cleanup can track and destroy it if needed
				if (a) a->resources.addImageView((*viewVec)[layer], "TextureArrayManager: layerView");
	}

	// create ImGui texture (descriptor set) for this view
	(*texVec)[layer] = (ImTextureID)ImGui_ImplVulkan_AddTexture(sampler, (*viewVec)[layer], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	return (*texVec)[layer];
}

// -----------------------------------------------------------------------------
// alpha-only descriptor helpers
// -----------------------------------------------------------------------------

ImTextureID TextureArrayManager::getImTextureAlpha(size_t layer, int map) {
    if (layer >= layerAmount) return 0;
    VulkanApp* a = this->appPtr;
    if (!a) return 0;
    VkDevice device = a->getDevice();

    std::vector<VkImageView>* viewVec = nullptr;
    std::vector<ImTextureID>* texVec = nullptr;
    TextureImage* src = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    switch (map) {
        case 0: viewVec = &albedoLayerViews; texVec = &albedoImTextures; src = &albedoArray; sampler = albedoSampler; break;
        case 1: viewVec = &normalLayerViews; texVec = &normalImTextures; src = &normalArray; sampler = normalSampler; break;
        case 2: viewVec = &bumpLayerViews; texVec = &bumpImTextures; src = &bumpArray; sampler = bumpSampler; break;
        case 3: viewVec = &roughnessLayerViews; texVec = &roughnessImTextures; src = &roughnessArray; sampler = roughnessSampler; break;
        case 4: viewVec = &aoLayerViews; texVec = &aoImTextures; src = &aoArray; sampler = aoSampler; break;
        default: return 0;
    }

    // ensure vectors are sized for new alpha views too (reuse same slots)
    if (viewVec->size() != layerAmount) viewVec->resize(layerAmount, VK_NULL_HANDLE);
    if (texVec->size() != layerAmount) texVec->resize(layerAmount, 0);

    if ((*texVec)[layer]) return (*texVec)[layer];

    // create swizzled view for alpha channel
    if (!(*viewVec)[layer]) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = src->image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_A;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_A;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_A;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = src->mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = static_cast<uint32_t>(layer);
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device, &viewInfo, nullptr, &(*viewVec)[layer]) != VK_SUCCESS) {
            return 0;
        }
        if (a) a->resources.addImageView((*viewVec)[layer], "TextureArrayManager: alphaLayerView");
    }

    (*texVec)[layer] = (ImTextureID)ImGui_ImplVulkan_AddTexture(sampler, (*viewVec)[layer], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return (*texVec)[layer];
}
