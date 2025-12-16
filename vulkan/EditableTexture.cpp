#include "EditableTexture.hpp"
#include "VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <backends/imgui_impl_vulkan.h>

void EditableTexture::init(VulkanApp* app, uint32_t w, uint32_t h, VkFormat fmt, const char* nm) {
	this->app = app;
	width = w;
	height = h;
	format = fmt;
	name = nm ? nm : "Editable Texture";
	bytesPerPixel = (format == VK_FORMAT_R8_UNORM) ? 1 : 4;
	cpuData.assign(width * height * bytesPerPixel, 0);
	isDirty = false;

	// Create GPU resources
	app->createImage(width, height, format, VK_IMAGE_TILING_OPTIMAL, 1,
					 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
					 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);

	// create view
	VkImageViewCreateInfo viewInfo = {};
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

	createImGuiDescriptor();
}

void EditableTexture::cleanup() {
	if (imguiDescSet != VK_NULL_HANDLE) {
		ImGui_ImplVulkan_RemoveTexture((VkDescriptorSet)imguiDescSet);
		imguiDescSet = VK_NULL_HANDLE;
	}
	if (sampler != VK_NULL_HANDLE) vkDestroySampler(app->getDevice(), sampler, nullptr);
	if (view != VK_NULL_HANDLE) vkDestroyImageView(app->getDevice(), view, nullptr);
	if (image != VK_NULL_HANDLE) vkDestroyImage(app->getDevice(), image, nullptr);
	if (memory != VK_NULL_HANDLE) vkFreeMemory(app->getDevice(), memory, nullptr);
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

void EditableTexture::updateGPU() {
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

	// destroy staging
	if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, staging.buffer, nullptr);
	if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device, staging.memory, nullptr);

	isDirty = false;
}

void EditableTexture::renderImGui() {
	if (imguiDescSet == VK_NULL_HANDLE) createImGuiDescriptor();
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
VkDescriptorSet EditableTexture::getImGuiDescriptorSet() const { return imguiDescSet; }
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
}

void EditableTexture::transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout) {
	app->transitionImageLayout(image, format, oldLayout, newLayout);
}

void EditableTexture::copyBufferToImage(VkBuffer buffer) {
	app->copyBufferToImage(buffer, image, width, height);
}
