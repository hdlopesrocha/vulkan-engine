// Minimal implementation file for TextureArrayManager
#include "TextureArrayManager.hpp"
#include "VulkanApp.hpp"
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>
#include "../vulkan/EditableTexture.hpp"

void TextureArrayManager::allocate(uint32_t layers, uint32_t w, uint32_t h) {
	layerAmount = layers;
	width = w;
	height = h;
}

// Helper to cleanup a TextureImage if it already has resources
static void cleanupTextureImage(VulkanApp* app, TextureImage &ti) {
	if (ti.view != VK_NULL_HANDLE) {
		vkDestroyImageView(app->getDevice(), ti.view, nullptr);
		ti.view = VK_NULL_HANDLE;
	}
	if (ti.image != VK_NULL_HANDLE) {
		vkDestroyImage(app->getDevice(), ti.image, nullptr);
		ti.image = VK_NULL_HANDLE;
	}
	if (ti.memory != VK_NULL_HANDLE) {
		vkFreeMemory(app->getDevice(), ti.memory, nullptr);
		ti.memory = VK_NULL_HANDLE;
	}
	ti.mipLevels = 1;
}

// Cleanup sampler if present
static void cleanupSampler(VulkanApp* app, VkSampler &s) {
	if (s != VK_NULL_HANDLE) {
		vkDestroySampler(app->getDevice(), s, nullptr);
		s = VK_NULL_HANDLE;
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
	for (auto &tex : albedoImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)tex);
	}
	for (auto &tex : normalImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)tex);
	}
	for (auto &tex : bumpImTextures) {
		if (tex && (VkDescriptorSet)tex != VK_NULL_HANDLE) ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)tex);
	}
	VkDevice device = app->getDevice();
	for (auto v : albedoLayerViews) if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
	for (auto v : normalLayerViews) if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
	for (auto v : bumpLayerViews) if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
	albedoLayerViews.clear(); normalLayerViews.clear(); bumpLayerViews.clear();
	albedoImTextures.clear(); normalImTextures.clear(); bumpImTextures.clear();
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

	// We'll create 2D array images with a single mip level for now
	uint32_t mipLevels = 1;

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
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

		// Transition to TRANSFER_DST_OPTIMAL then to SHADER_READ_ONLY_OPTIMAL
		// VulkanApp::transitionImageLayout supports these two-step transitions.
		app->transitionImageLayout(out.image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		app->transitionImageLayout(out.image, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	};

	// Albedo uses sRGB format
	createArray(albedoArray, VK_FORMAT_R8G8B8A8_SRGB, true);
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

uint TextureArrayManager::load(char* albedoFile, char* normalFile, char* bumpFile) {
	if (!app) throw std::runtime_error("TextureArrayManager::load: manager has no VulkanApp (call allocate(..., app) first)");
	if (layerAmount == 0) throw std::runtime_error("TextureArrayManager::load: layerAmount == 0");
	if (currentLayer >= layerAmount) throw std::runtime_error("TextureArrayManager::load: currentLayer >= layerAmount");

	VulkanApp* a = this->app;
	VkDevice device = a->getDevice();

	struct Img { char* path; TextureImage* dstImage; VkFormat format; bool srgb; } imgs[3] = {
		{ albedoFile, &albedoArray, VK_FORMAT_R8G8B8A8_SRGB, true },
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
		barrier1.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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

		// Transition back to SHADER_READ_ONLY_OPTIMAL
		VkImageMemoryBarrier barrier2{};
		barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier2.image = imgs[i].dstImage->image;
		barrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier2.subresourceRange.baseMipLevel = 0;
		barrier2.subresourceRange.levelCount = imgs[i].dstImage->mipLevels;
		barrier2.subresourceRange.baseArrayLayer = currentLayer;
		barrier2.subresourceRange.layerCount = 1;
		barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
							 0, nullptr, 0, nullptr, 1, &barrier2);

		a->endSingleTimeCommands(cmd);

		// cleanup staging
		if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging.buffer, nullptr);
		if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device, staging.memory, nullptr);
	}

	// increment current layer
	return currentLayer++;
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

		a->endSingleTimeCommands(cmd);
	}

	// cleanup staging
	if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging.buffer, nullptr);
	if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device, staging.memory, nullptr);

	return currentLayer++;
}

void TextureArrayManager::updateLayerFromEditable(uint32_t layer, const EditableTexture& tex) {
	// Keep backward-compatible behavior (copy into albedo array)
	updateLayerFromEditableMap(layer, tex, 0);
}

void TextureArrayManager::updateLayerFromEditableMap(uint32_t layer, const EditableTexture& tex, int map) {
	if (!app) throw std::runtime_error("TextureArrayManager::updateLayerFromEditableMap: no VulkanApp");
	if (layer >= layerAmount) throw std::runtime_error("TextureArrayManager::updateLayerFromEditableMap: layer out of range");

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

	// Copy into selected array layer
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
		if (map == 0) viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		else viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
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
