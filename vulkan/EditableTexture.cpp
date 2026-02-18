#include "EditableTexture.hpp"
#include "VulkanApp.hpp"
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>

void EditableTexture::init(VulkanApp* app, uint32_t w, uint32_t h, VkFormat fmt, const char* nm) {
	width = w;
	height = h;
	format = fmt;
	name = nm ? nm : "Editable Texture";
	bytesPerPixel = (format == VK_FORMAT_R8_UNORM) ? 1 : 4;
    printf("[EditableTexture::init] name='%s' w=%u h=%u bytes=%u\n", name.c_str(), width, height, bytesPerPixel);

    cpuData.assign((size_t)width * (size_t)height * (size_t)bytesPerPixel, 0);
    isDirty = false;

	// Create GPU resources (allow transfer-src so we can copy from this image)
	app->createImage(width, height, format, VK_IMAGE_TILING_OPTIMAL, 1,
					 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
					 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);

    // create view
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &view) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image view");
	}
    printf("[EditableTexture] createImageView: view=%p image=%p format=%d\n", (void*)view, (void*)image, (int)format);
	// Register image view
	app->resources.addImageView(view, "EditableTexture: view");

	// sampler
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = 16.0f;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler");
	}
    printf("[EditableTexture] createSampler: sampler=%p\n", (void*)sampler);
	// Register sampler
	app->resources.addSampler(sampler, "EditableTexture: sampler");

	createImGuiDescriptor();
}

void EditableTexture::cleanup() {
	if (imguiDescSet != VK_NULL_HANDLE) {
		ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)imguiDescSet);
		imguiDescSet = VK_NULL_HANDLE;
	}
	// Do not destroy Vulkan objects here; central manager handles destruction.
	sampler = VK_NULL_HANDLE;
	view = VK_NULL_HANDLE;
	image = VK_NULL_HANDLE;
	memory = VK_NULL_HANDLE;
	cpuData.clear();
}

void EditableTexture::setPixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	if (x >= width || y >= height) return;
	size_t idx = (y * width + x) * bytesPerPixel;
	if (bytesPerPixel >= 4) {
		cpuData[idx + 0] = r;
		cpuData[idx + 1] = g;
		cpuData[idx + 2] = b;
		cpuData[idx + 3] = a;
	}
	isDirty = true;
}

void EditableTexture::setPixelGray(uint32_t x, uint32_t y, uint8_t value) {
	if (x >= width || y >= height) return;
	size_t idx = (y * width + x) * bytesPerPixel;
	if (bytesPerPixel == 1) {
		cpuData[idx] = value;
	}
	isDirty = true;
}

void EditableTexture::fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
	if (bytesPerPixel == 1) {
		std::fill(cpuData.begin(), cpuData.end(), r);
	} else {
		for (size_t i = 0; i < width * height; ++i) {
			size_t idx = i * bytesPerPixel;
			cpuData[idx + 0] = r;
			cpuData[idx + 1] = g;
			cpuData[idx + 2] = b;
			cpuData[idx + 3] = a;
		}
	}
	isDirty = true;
}

void EditableTexture::updateGPU(VulkanApp* app) {
	if (!isDirty) return;
	VkDevice device = app->getDevice();
	size_t bufSize = cpuData.size();
	if (bufSize == 0) return;

	Buffer staging = app->createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	void* data;
	vkMapMemory(device, staging.memory, 0, bufSize, 0, &data);
	memcpy(data, cpuData.data(), bufSize);
	vkUnmapMemory(device, staging.memory);

	app->transitionImageLayout(image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	app->copyBufferToImage(staging.buffer, image, width, height);
	app->transitionImageLayout(image, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	// Defer actual destruction to VulkanResourceManager; clear local handles
	if (staging.buffer != VK_NULL_HANDLE) staging.buffer = VK_NULL_HANDLE;
	if (staging.memory != VK_NULL_HANDLE) staging.memory = VK_NULL_HANDLE;

	isDirty = false;
}

void EditableTexture::renderImGui() {
	if (imguiDescSet != VK_NULL_HANDLE) createImGuiDescriptor();
	if (imguiDescSet != VK_NULL_HANDLE) {
		ImGui::Image((ImTextureID)imguiDescSet, ImVec2((float)width, (float)height));
	}
}

VkImageView EditableTexture::getView() const { return view; }
VkSampler EditableTexture::getSampler() const { return sampler; }
VkImage EditableTexture::getImage() const { return image; }
bool EditableTexture::getDirty() const { return isDirty; }
uint32_t EditableTexture::getWidth() const { return width; }
uint32_t EditableTexture::getHeight() const { return height; }
uint32_t EditableTexture::getBytesPerPixel() const { return bytesPerPixel; }
VkDescriptorSet EditableTexture::getImGuiDescriptorSet() {
    if (imguiDescSet == VK_NULL_HANDLE) createImGuiDescriptor();
    return imguiDescSet;
}
const uint8_t* EditableTexture::getPixelData() const { return cpuData.empty() ? nullptr : cpuData.data(); }

TextureImage EditableTexture::getTextureImage() const {
	TextureImage ti;
	ti.image = image;
	ti.memory = memory;
	ti.view = view;
	ti.mipLevels = 1;
	return ti;
}

void EditableTexture::createImGuiDescriptor() {
	if (imguiDescSet != VK_NULL_HANDLE) return;
	ImTextureID id = ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	imguiDescSet = (VkDescriptorSet)id;
	if (imguiDescSet == VK_NULL_HANDLE) {
		printf("[EditableTexture] Failed to create ImGui descriptor for '%s'\n", name.c_str());
	} else {
		printf("[EditableTexture] Created ImGui descriptor %p for '%s'\n", (void*)imguiDescSet, name.c_str());
	}
}

void EditableTexture::transitionImageLayout(VulkanApp* app, VkImageLayout oldLayout, VkImageLayout newLayout) {
	app->transitionImageLayout(image, format, oldLayout, newLayout);
}

void EditableTexture::copyBufferToImage(VulkanApp* app, VkBuffer buffer) {
	app->copyBufferToImage(buffer, image, width, height);
}

void EditableTexture::debugDumpFirstPixel(VulkanApp* app) {
	// Create a staging buffer large enough for one row
	VkDevice device = app->getDevice();
	size_t pixelSize = bytesPerPixel;
	size_t bufSize = pixelSize * width * height; // read whole image for simplicity

	Buffer staging = app->createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	VkCommandBuffer cmd = app->beginSingleTimeCommands();

	// Transition image to TRANSFER_SRC
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = {0, 0, 0};
	region.imageExtent = { width, height, 1 };

	vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1, &region);

	// Transition back
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	app->endSingleTimeCommands(cmd);

	// Map and read first pixel
	void* data;
	vkMapMemory(device, staging.memory, 0, pixelSize, 0, &data);
	uint8_t *bytes = reinterpret_cast<uint8_t*>(data);
	if (bytesPerPixel == 4) {
		printf("[EditableTexture] First pixel RGBA = %u %u %u %u\n", bytes[0], bytes[1], bytes[2], bytes[3]);
	} else if (bytesPerPixel == 1) {
		printf("[EditableTexture] First pixel R = %u\n", bytes[0]);
	} else {
		printf("[EditableTexture] First pixel raw bytes: ");
		for (size_t i=0;i<pixelSize;i++) printf("%02x ", bytes[i]);
		printf("\n");
	}
	vkUnmapMemory(device, staging.memory);

	// Defer actual destruction to VulkanResourceManager; clear local handles
	if (staging.buffer != VK_NULL_HANDLE) staging.buffer = VK_NULL_HANDLE;
	if (staging.memory != VK_NULL_HANDLE) staging.memory = VK_NULL_HANDLE;
}
