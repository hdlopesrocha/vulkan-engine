// Minimal implementation file for TextureArrayManager
#include "TextureArrayManager.hpp"
#include "VulkanApp.hpp"
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>
#include "../vulkan/EditableTexture.hpp"
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
            if (v < 0) v = 0; if (v > 255) v = 255;
            p[c] = static_cast<unsigned char>(v);
        }
        // alpha channel left as-is
    }
}

void TextureArrayManager::allocate(uint32_t layers, uint32_t w, uint32_t h) {
	layerAmount = layers;
	width = w;
	height = h;
}

void TextureArrayManager::notifyAllocationListeners() {
	std::vector<std::function<void()>> listenersCopy;
	listenersCopy.reserve(allocationListeners.size());
	for (auto &l : allocationListeners) if (l) listenersCopy.push_back(l);
	for (size_t i = 0; i < listenersCopy.size(); ++i) {
		try {
			listenersCopy[i]();
		} catch (const std::exception &e) {
			fprintf(stderr, "[TextureArrayManager] allocation listener %zu threw std::exception: %s\n", i, e.what());
		} catch (...) {
			fprintf(stderr, "[TextureArrayManager] allocation listener %zu threw unknown exception\n", i);
		}
	}
}

// Helper to cleanup a TextureImage if it already has resources
static void cleanupTextureImage(VulkanApp* app, TextureImage &ti) {
	if (!app) return;
	VkDevice device = app->getDevice();
	// Destroy view
	if (ti.view != VK_NULL_HANDLE) {
		VkImageView v = ti.view;
		if (app->hasPendingCommandBuffers()) {
			// Schedule destroy after pending commands complete
			app->deferDestroyUntilAllPending([device, v](){ vkDestroyImageView(device, v, nullptr); });
			ti.view = VK_NULL_HANDLE;
		} else {
			vkDestroyImageView(device, ti.view, nullptr);
			ti.view = VK_NULL_HANDLE;
		}
	}
	// Destroy image
	if (ti.image != VK_NULL_HANDLE) {
		VkImage img = ti.image;
		if (app->hasPendingCommandBuffers()) {
			app->deferDestroyUntilAllPending([device, img](){ vkDestroyImage(device, img, nullptr); });
			ti.image = VK_NULL_HANDLE;
		} else {
			vkDestroyImage(device, ti.image, nullptr);
			ti.image = VK_NULL_HANDLE;
		}
	}
	// Free memory
	if (ti.memory != VK_NULL_HANDLE) {
		VkDeviceMemory mem = ti.memory;
		if (app->hasPendingCommandBuffers()) {
			app->deferDestroyUntilAllPending([device, mem](){ vkFreeMemory(device, mem, nullptr); });
			ti.memory = VK_NULL_HANDLE;
		} else {
			vkFreeMemory(device, ti.memory, nullptr);
			ti.memory = VK_NULL_HANDLE;
		}
	}
	ti.mipLevels = 1;
}

// Cleanup sampler if present
static void cleanupSampler(VulkanApp* app, VkSampler &s) {
	if (!app) return;
	VkDevice device = app->getDevice();
	if (s != VK_NULL_HANDLE) {
		VkSampler ss = s;
		if (app->hasPendingCommandBuffers()) {
			app->deferDestroyUntilAllPending([device, ss](){ vkDestroySampler(device, ss, nullptr); });
			s = VK_NULL_HANDLE;
		} else {
			vkDestroySampler(device, s, nullptr);
			s = VK_NULL_HANDLE;
		}
	}
}

void TextureArrayManager::destroy(VulkanApp* app) {
	if (!app) return;
	cleanupTextureImage(app, albedoArray);
	cleanupTextureImage(app, normalArray);
	cleanupTextureImage(app, bumpArray);
	cleanupSampler(app, albedoSampler);
	cleanupSampler(app, normalSampler);
	cleanupSampler(app, bumpSampler);
	// Remove any ImGui textures and destroy per-layer views
	// Remove any ImGui textures; if async work is pending defer removal to avoid destroying descriptor sets in use
	for (auto &tex : albedoImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
				tex = nullptr;
			} else {
				ImGui_ImplVulkan_RemoveTexture(ds);
				tex = nullptr;
			}
		}
	}
	for (auto &tex : normalImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
				tex = nullptr;
			} else {
				ImGui_ImplVulkan_RemoveTexture(ds);
				tex = nullptr;
			}
		}
	}
	for (auto &tex : bumpImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)tex;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
				tex = nullptr;
			} else {
				ImGui_ImplVulkan_RemoveTexture(ds);
				tex = nullptr;
			}
		}
	}
	VkDevice device = app->getDevice();
	// Destroy per-layer views; defer if async submissions are active
	for (auto &v : albedoLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([device, iv](){ vkDestroyImageView(device, iv, nullptr); });
				v = VK_NULL_HANDLE;
			} else {
				vkDestroyImageView(device, v, nullptr);
				v = VK_NULL_HANDLE;
			}
		}
	}
	for (auto &v : normalLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([device, iv](){ vkDestroyImageView(device, iv, nullptr); });
				v = VK_NULL_HANDLE;
			} else {
				vkDestroyImageView(device, v, nullptr);
				v = VK_NULL_HANDLE;
			}
		}
	}
	for (auto &v : bumpLayerViews) {
		if (v != VK_NULL_HANDLE) {
			VkImageView iv = v;
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([device, iv](){ vkDestroyImageView(device, iv, nullptr); });
				v = VK_NULL_HANDLE;
			} else {
				vkDestroyImageView(device, v, nullptr);
				v = VK_NULL_HANDLE;
			}
		}
	}
	albedoLayerViews.clear(); normalLayerViews.clear(); bumpLayerViews.clear();
	albedoImTextures.clear(); normalImTextures.clear(); bumpImTextures.clear();
	// bump version to indicate array resources were destroyed
	++this->version;
	// notify listeners that arrays were destroyed
	notifyAllocationListeners();
}

void TextureArrayManager::allocate(uint32_t layers, uint32_t w, uint32_t h, VulkanApp* app) {
	if (!app) throw std::runtime_error("TextureArrayManager::allocate: app is null");

	layerAmount = layers;
	width = w;
	height = h;

	// destroy previous resources if present
	cleanupTextureImage(app, albedoArray);
	cleanupTextureImage(app, normalArray);
	cleanupTextureImage(app, bumpArray);

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
		// Allow sampling and transfers, and also allow storage usage so compute
		// shaders can write directly into array layers via imageStore.
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

		VkDevice device = app->getDevice();

		if (vkCreateImage(device, &imageInfo, nullptr, &out.image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture array image");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, out.image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = app->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &out.memory) != VK_SUCCESS) {
			vkDestroyImage(device, out.image, nullptr);
			out.image = VK_NULL_HANDLE;
			throw std::runtime_error("failed to allocate texture array memory");
		}

		vkBindImageMemory(device, out.image, out.memory, 0);

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
			vkDestroyImage(device, out.image, nullptr);
			vkFreeMemory(device, out.memory, nullptr);
			out.image = VK_NULL_HANDLE;
			out.memory = VK_NULL_HANDLE;
			throw std::runtime_error("failed to create texture array image view");
		}

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

	// cleanup existing samplers and create new ones
	cleanupSampler(app, albedoSampler);
	cleanupSampler(app, normalSampler);
	cleanupSampler(app, bumpSampler);

	albedoSampler = app->createTextureSampler(mipLevels);
	normalSampler = app->createTextureSampler(mipLevels);
	bumpSampler = app->createTextureSampler(mipLevels);
	// store app reference for later load() calls
	this->app = app;
	// initialize layer initialized flags
	layerInitialized.clear();
	layerInitialized.resize(layerAmount, 0);

	// initialize per-layer layout trackers to UNDEFINED by default
	albedoLayerLayouts.clear(); albedoLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	normalLayerLayouts.clear(); normalLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);
	bumpLayerLayouts.clear(); bumpLayerLayouts.resize(layerAmount, VK_IMAGE_LAYOUT_UNDEFINED);

	// After creating arrays we transitioned all mips/layers to SHADER_READ_ONLY_OPTIMAL above; reflect that state
	for (uint32_t i = 0; i < layerAmount; ++i) {
		albedoLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		normalLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		bumpLayerLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}

	// bump version so users can detect reallocation of GPU resources
	++this->version;
	// notify listeners that arrays changed
	notifyAllocationListeners();
}

// Nearest-neighbor resize (RGBA8)
static unsigned char* resizeNearest(const unsigned char* src, int srcW, int srcH, int dstW, int dstH) {
	unsigned char* dst = new unsigned char[static_cast<size_t>(dstW) * dstH * 4];
	for (int y = 0; y < dstH; ++y) {
		int sy = (y * srcH) / dstH;
		for (int x = 0; x < dstW; ++x) {
			int sx = (x * srcW) / dstW;
			const unsigned char* s = &src[(sy * srcW + sx) * 4];
			unsigned char* d = &dst[(y * dstW + x) * 4];
			d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
		}
	}
	return dst;
}

uint TextureArrayManager::load(const char* albedoFile, const char* normalFile, const char* bumpFile) {
	if (!app) throw std::runtime_error("TextureArrayManager::load: manager has no VulkanApp (call allocate(..., app) first)");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::load: layerAmount == 0");
	if (currentLayer >= layerAmount) throw std::runtime_error("TextureArrayManager::load: currentLayer >= layerAmount");

	VulkanApp* a = this->app;
	VkDevice device = a->getDevice();
	std::cout << "[TextureArrayManager] Loading textures into layer " << currentLayer << ": "
			  << (albedoFile ? albedoFile : "(none)") << ", "
			  << (normalFile ? normalFile : "(none)") << ", "
			  << (bumpFile ? bumpFile : "(none)") << std::endl;
	struct Img { const char* path; TextureImage* dstImage; VkFormat format; bool srgb; } imgs[3] = {
		{ albedoFile, &albedoArray, VK_FORMAT_R8G8B8A8_UNORM, true },
		{ normalFile, &normalArray, VK_FORMAT_R8G8B8A8_UNORM, false },
		{ bumpFile,   &bumpArray,   VK_FORMAT_R8G8B8A8_UNORM, false }
	};

	for (int i = 0; i < 3; ++i) {
		if (!imgs[i].path) continue;
		int texW=0, texH=0, texC=0;
		unsigned char* pixels = stbi_load(imgs[i].path, &texW, &texH, &texC, 4);
		if (!pixels) {
			throw std::runtime_error(std::string("failed to load texture: ") + imgs[i].path);
		}

		unsigned char* uploadData = pixels;
		bool resized = false;
		if (texW != static_cast<int>(width) || texH != static_cast<int>(height)) {
			unsigned char* r = resizeNearest(pixels, texW, texH, static_cast<int>(width), static_cast<int>(height));
			uploadData = r;
			resized = true;
		}

		// If this image is flagged as sRGB (color/albedo), convert to linear before storing as UNORM
		if (imgs[i].srgb) {
			convertSRGB8ToLinearInPlace(uploadData, static_cast<size_t>(width) * static_cast<size_t>(height));
		}

		VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;
		Buffer staging = a->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		void* data;
		vkMapMemory(device, staging.memory, 0, imageSize, 0, &data);
		memcpy(data, uploadData, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, staging.memory);

		if (resized) delete[] uploadData;
		stbi_image_free(pixels);

		// Build command buffer and perform per-layer copy with barriers
		VkCommandBuffer cmd = a->beginSingleTimeCommands();

		// Transition this layer to TRANSFER_DST_OPTIMAL
		VkImageMemoryBarrier barrier1{};
		barrier1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	// Use current tracked layout as oldLayout so transitions are correct. Clamp
	// transient TRANSFER_DST tracked states to SHADER_READ_ONLY to avoid
	// validation mismatches where the validation layer's known layout differs
	// due to concurrency or delayed updates.
	VkImageLayout currentLayout = getLayerLayout(i, currentLayer);
	if (currentLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	}
	barrier1.oldLayout = currentLayout;
	barrier1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier1.image = imgs[i].dstImage->image;
	barrier1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier1.subresourceRange.baseMipLevel = 0;
	barrier1.subresourceRange.levelCount = imgs[i].dstImage->mipLevels;
	barrier1.subresourceRange.baseArrayLayer = currentLayer;
	barrier1.subresourceRange.layerCount = 1;
	barrier1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				 0, nullptr, 0, nullptr, 1, &barrier1);
	// reflect the pending state
	setLayerLayout(i, currentLayer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = currentLayer;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = {0,0,0};
	region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(cmd, staging.buffer, imgs[i].dstImage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// finish the short-lived command buffer so the copy is flushed
		a->endSingleTimeCommands(cmd);

		// Generate mipmaps for the uploaded layer (this will transition mip levels to correct layouts)
		if (imgs[i].dstImage->mipLevels > 1) {
			a->generateMipmaps(imgs[i].dstImage->image, imgs[i].format, static_cast<int32_t>(width), static_cast<int32_t>(height), imgs[i].dstImage->mipLevels, 1, currentLayer);
			// mark layer layout as SHADER_READ_ONLY_OPTIMAL after mipgen
			setLayerLayout(i, currentLayer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		} else {
			// single-level: transition layer to SHADER_READ_ONLY_OPTIMAL
			VkCommandBuffer cmd2 = a->beginSingleTimeCommands();
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = imgs[i].dstImage->image;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = imgs[i].dstImage->mipLevels;
			barrier.subresourceRange.baseArrayLayer = currentLayer;
			barrier.subresourceRange.layerCount = 1;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd2, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
					 0, nullptr, 0, nullptr, 1, &barrier);
			a->endSingleTimeCommands(cmd2);
			// set tracked layout
			setLayerLayout(i, currentLayer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging.buffer, nullptr);
		if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device, staging.memory, nullptr);
	}

	// increment current layer
	setLayerInitialized(currentLayer, true);
	return currentLayer++;
}

size_t TextureArrayManager::loadTriples(const std::vector<TextureTriple> &triples) {
	if (!app) throw std::runtime_error("TextureArrayManager::loadTriples: manager has no VulkanApp");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::loadTriples: layerAmount == 0");
	size_t loaded = 0;
	for (const auto &t : triples) {
		if (currentLayer >= layerAmount) {
			fprintf(stderr, "[TextureArrayManager] Reached texture array capacity (%u layers)\n", layerAmount);
			break;
		}
		try {
			load(t.albedo, t.normal, t.bump);
			++loaded;
		} catch (const std::exception &e) {
			fprintf(stderr, "[TextureArrayManager] Failed to load %s: %s\n", t.albedo ? t.albedo : "(null)", e.what());
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
		default: return VK_IMAGE_LAYOUT_UNDEFINED;
	}
}

void TextureArrayManager::setLayerLayout(int map, uint32_t layer, VkImageLayout layout) {
	if (layer >= layerAmount) return;
	switch (map) {
		case 0: if (albedoLayerLayouts.size() > layer) albedoLayerLayouts[layer] = layout; break;
		case 1: if (normalLayerLayouts.size() > layer) normalLayerLayouts[layer] = layout; break;
		case 2: if (bumpLayerLayouts.size() > layer) bumpLayerLayouts[layer] = layout; break;
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

uint TextureArrayManager::create() {
	if (!app) throw std::runtime_error("TextureArrayManager::create: manager has no VulkanApp (call allocate(..., app) first)");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::create: layerAmount == 0");
	if (currentLayer >= layerAmount) throw std::runtime_error("TextureArrayManager::create: currentLayer >= layerAmount");

	VulkanApp* a = this->app;
	VkDevice device = a->getDevice();

	VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

	// create a zeroed staging buffer
	Buffer staging = a->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	void* mapped = nullptr;
	vkMapMemory(device, staging.memory, 0, imageSize, 0, &mapped);
	memset(mapped, 0, static_cast<size_t>(imageSize));
	vkUnmapMemory(device, staging.memory);

	TextureImage* imgs[3] = { &albedoArray, &normalArray, &bumpArray };

	for (int i = 0; i < 3; ++i) {
		TextureImage* dst = imgs[i];
		if (!dst || dst->image == VK_NULL_HANDLE) continue;

		VkCommandBuffer cmd = a->beginSingleTimeCommands();

		// transition layer to TRANSFER_DST_OPTIMAL
		VkImageMemoryBarrier barrier1{};
		barrier1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier1.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier1.image = dst->image;
		barrier1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier1.subresourceRange.baseMipLevel = 0;
		barrier1.subresourceRange.levelCount = dst->mipLevels;
		barrier1.subresourceRange.baseArrayLayer = currentLayer;
		barrier1.subresourceRange.layerCount = 1;
		barrier1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
							 0, nullptr, 0, nullptr, 1, &barrier1);

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = currentLayer;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = {0,0,0};
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(cmd, staging.buffer, dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// Only transition to SHADER_READ_ONLY if we're NOT generating mipmaps
		// (generateMipmaps handles the transition itself)
		if (dst->mipLevels <= 1) {
			VkImageMemoryBarrier barrier2{};
			barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier2.image = dst->image;
			barrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier2.subresourceRange.baseMipLevel = 0;
			barrier2.subresourceRange.levelCount = dst->mipLevels;
			barrier2.subresourceRange.baseArrayLayer = currentLayer;
			barrier2.subresourceRange.layerCount = 1;
			barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
								 0, nullptr, 0, nullptr, 1, &barrier2);
		}

		a->endSingleTimeCommands(cmd);
		// Generate mipmaps for this layer if necessary (this also transitions to SHADER_READ_ONLY)
		if (dst->mipLevels > 1) {
			a->generateMipmaps(dst->image, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), dst->mipLevels, 1, currentLayer);
		}	}

	// cleanup staging
	if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging.buffer, nullptr);
	if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device, staging.memory, nullptr);

	setLayerInitialized(currentLayer, true);
	return currentLayer++;
}

void TextureArrayManager::updateLayerFromEditable(uint32_t layer, const EditableTexture& tex) {
	// Keep backward-compatible behavior (copy into albedo array)
	updateLayerFromEditableMap(layer, tex, 0);
}

#include "TextureMixer.hpp"

void TextureArrayManager::updateLayerFromEditableMap(uint32_t layer, const EditableTexture& tex, int map) {
	if (!app) throw std::runtime_error("TextureArrayManager::updateLayerFromEditableMap: no VulkanApp");
	if (layer >= layerAmount) throw std::runtime_error("TextureArrayManager::updateLayerFromEditableMap: layer out of range");

	// If a per-layer generation is in-flight for this layer, wait for it to finish to avoid layout races
	TextureMixer* gm = TextureMixer::getGlobalInstance();
	if (gm && gm->isLayerGenerationPending(layer)) {
		fprintf(stderr, "[TextureArrayManager] Waiting for pending generation on layer %u before copying\n", layer);
		gm->waitForLayerGeneration(layer);
	}

	VulkanApp* a = this->app;
	VkDevice device = a->getDevice();

	VkImage srcImage = tex.getImage();
	VkImage dstImage = VK_NULL_HANDLE;
	if (map == 0) dstImage = albedoArray.image;
	else if (map == 1) dstImage = normalArray.image;
	else if (map == 2) dstImage = bumpArray.image;
	else throw std::runtime_error("TextureArrayManager::updateLayerFromEditableMap: invalid map");

	// Transition dst layer to TRANSFER_DST_OPTIMAL for the selected array image and src to TRANSFER_SRC_OPTIMAL
	VkCommandBuffer cmd = a->beginSingleTimeCommands();

	auto doBarrier = [&](VkImage dst, VkImageLayout oldDst, VkImageLayout newDst, uint32_t dstBaseLayer) {
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldDst;
		barrier.newLayout = newDst;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = dst;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = dstBaseLayer;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					  0, nullptr, 0, nullptr, 1, &barrier);
	};

	// source barrier: shader read -> transfer src
	VkImageMemoryBarrier srcBarrier{};
	srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	srcBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	srcBarrier.image = srcImage;
	srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	srcBarrier.subresourceRange.baseMipLevel = 0;
	srcBarrier.subresourceRange.levelCount = 1;
	srcBarrier.subresourceRange.baseArrayLayer = 0;
	srcBarrier.subresourceRange.layerCount = 1;
	srcBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
					  0, nullptr, 0, nullptr, 1, &srcBarrier);

	// Copy into selected array layer (ensure we log what we're doing)
	fprintf(stderr, "[TextureArrayManager] Copying into array image %p layer=%u\n", (void*)dstImage, layer);
	doBarrier(dstImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layer);
	VkImageCopy copyRegion{};
	copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.srcSubresource.mipLevel = 0;
	copyRegion.srcSubresource.baseArrayLayer = 0;
	copyRegion.srcSubresource.layerCount = 1;
	copyRegion.srcOffset = {0,0,0};
	copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.dstSubresource.mipLevel = 0;
	copyRegion.dstSubresource.baseArrayLayer = layer;
	copyRegion.dstSubresource.layerCount = 1;
	copyRegion.dstOffset = {0,0,0};
	copyRegion.extent = { width, height, 1 };
	vkCmdCopyImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

	// Transition dst layer back to SHADER_READ_ONLY_OPTIMAL
	fprintf(stderr, "[TextureArrayManager] Transitioning dst layer %u back to SHADER_READ_ONLY_OPTIMAL\n", layer);
	VkImageMemoryBarrier dstBack{};
	dstBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	dstBack.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	dstBack.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	dstBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	dstBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	dstBack.image = dstImage;
	dstBack.subresourceRange.baseMipLevel = 0;
	dstBack.subresourceRange.levelCount = 1;
	dstBack.subresourceRange.baseArrayLayer = layer;
	dstBack.subresourceRange.layerCount = 1;
	dstBack.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	dstBack.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
						 0, nullptr, 0, nullptr, 1, &dstBack);

	// Transition src back to SHADER_READ_ONLY_OPTIMAL
	VkImageMemoryBarrier srcBack = srcBarrier;
	srcBack.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	srcBack.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	srcBack.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	srcBack.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
						 0, nullptr, 0, nullptr, 1, &srcBack);

	a->endSingleTimeCommands(cmd);

		// Generate mipmaps for the updated layer if needed
		{
			uint32_t mipLevels = 1;
			switch (map) {
				case 0: mipLevels = albedoArray.mipLevels; break;
				case 1: mipLevels = normalArray.mipLevels; break;
				case 2: mipLevels = bumpArray.mipLevels; break;
			}
			if (mipLevels > 1) {
				a->generateMipmaps(dstImage, VK_FORMAT_R8G8B8A8_UNORM, static_cast<int32_t>(width), static_cast<int32_t>(height), mipLevels, 1, layer);
			}
		}
		std::vector<ImTextureID>* texVec = nullptr;
		auto viewVec = &albedoLayerViews;
		VkSampler sampler = albedoSampler;
		switch (map) {
			case 0: texVec = &albedoImTextures; viewVec = &albedoLayerViews; sampler = albedoSampler; break;
			case 1: texVec = &normalImTextures; viewVec = &normalLayerViews; sampler = normalSampler; break;
			case 2: texVec = &bumpImTextures; viewVec = &bumpLayerViews; sampler = bumpSampler; break;
		}

		if (texVec) {
			if (texVec->size() != layerAmount) texVec->resize(layerAmount, nullptr);
			if (viewVec->size() != layerAmount) viewVec->resize(layerAmount, VK_NULL_HANDLE);

			// If an ImGui descriptor exists for this layer, remove it so we can recreate a fresh one
			if ((*texVec)[layer] && (VkDescriptorSet)(*texVec)[layer] != VK_NULL_HANDLE) {
			VkDescriptorSet ds = (VkDescriptorSet)(*texVec)[layer];
			if (app && app->hasPendingCommandBuffers()) {
				app->deferDestroyUntilAllPending([ds](){ ImGui_ImplVulkan_RemoveTexture(ds); });
				(*texVec)[layer] = nullptr;
			} else {
				ImGui_ImplVulkan_RemoveTexture(ds);
				(*texVec)[layer] = nullptr;
			}
			}

			// Ensure a per-layer view exists (create if missing)
			if (!(*viewVec)[layer]) {
				TextureImage* arrImg = nullptr;
				switch (map) {
					case 0: arrImg = &albedoArray; break;
					case 1: arrImg = &normalArray; break;
					case 2: arrImg = &bumpArray; break;
				}
				if (arrImg && arrImg->image != VK_NULL_HANDLE) {
					VkImageViewCreateInfo viewInfo{};
					viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
					viewInfo.image = arrImg->image;
					viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
				if (map == 0) viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
					else viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
					viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					viewInfo.subresourceRange.baseMipLevel = 0;
					viewInfo.subresourceRange.levelCount = arrImg->mipLevels;
					viewInfo.subresourceRange.baseArrayLayer = static_cast<uint32_t>(layer);
					viewInfo.subresourceRange.layerCount = 1;
					if (vkCreateImageView(device, &viewInfo, nullptr, &(*viewVec)[layer]) != VK_SUCCESS) {
						// Failed to create view; leave tex as nullptr
					}
				}
			}

			// Create new ImGui descriptor for this layer
			if (!(*texVec)[layer] && (*viewVec)[layer] && sampler != VK_NULL_HANDLE) {
				ImTextureID id = ImGui_ImplVulkan_AddTexture(sampler, (*viewVec)[layer], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				(*texVec)[layer] = id;
			}
		}
	}

bool TextureArrayManager::isLayerInitialized(uint32_t layer) const {
	if (layer >= layerInitialized.size()) return false;
	return layerInitialized[layer] != 0;
}

void TextureArrayManager::setLayerInitialized(uint32_t layer, bool v) {
	if (layer >= layerInitialized.size()) return;
	layerInitialized[layer] = v ? 1 : 0;
}

ImTextureID TextureArrayManager::getImTexture(size_t layer, int map) {
	if (layer >= layerAmount) return nullptr;
	VulkanApp* a = this->app;
	if (!a) return nullptr;
	VkDevice device = a->getDevice();

	std::vector<VkImageView>* viewVec = nullptr;
	std::vector<ImTextureID>* texVec = nullptr;
	TextureImage* src = nullptr;
	VkSampler sampler = VK_NULL_HANDLE;
	switch (map) {
		case 0: viewVec = &albedoLayerViews; texVec = &albedoImTextures; src = &albedoArray; sampler = albedoSampler; break;
		case 1: viewVec = &normalLayerViews; texVec = &normalImTextures; src = &normalArray; sampler = normalSampler; break;
		case 2: viewVec = &bumpLayerViews; texVec = &bumpImTextures; src = &bumpArray; sampler = bumpSampler; break;
		default: return nullptr;
	}

	// ensure vectors are sized
	if (viewVec->size() != layerAmount) viewVec->resize(layerAmount, VK_NULL_HANDLE);
	if (texVec->size() != layerAmount) texVec->resize(layerAmount, nullptr);

	// if ImTextureID already created, return it
	if ((*texVec)[layer]) return (*texVec)[layer];

	// create a per-layer 2D image view
	if (!(*viewVec)[layer]) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = src->image;
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
			return nullptr;
		}
	}

	// create ImGui texture (descriptor set) for this view
	ImTextureID id = ImGui_ImplVulkan_AddTexture(sampler, (*viewVec)[layer], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	(*texVec)[layer] = id;
	return id;
}
