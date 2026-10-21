#include "VideoTextureService.hpp"

#include "../vulkan/core/VulkanApp.hpp"
#include "../vulkan/pipeline/VideoConvertPushConstants.hpp"
#include "../vulkan/resources/Buffer.hpp"
#include "../vulkan/resources/TextureArrayManager.hpp"

#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#ifdef HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#endif

namespace {

using Clock = std::chrono::steady_clock;
double nowMs() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

std::string lowerExtension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

std::string envString(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

bool envFlag(const char* name) {
    const std::string v = envString(name);
    return !v.empty() && v != "0";
}

int64_t fileBytes(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<int64_t>(size);
}

}  // namespace

#ifdef HAVE_FFMPEG

namespace {

// rgb = matrix * (Y, U, V) + offset, derived from the stream's colour metadata
// so the compute shader stays codec-agnostic: BT.601/709/2020 with limited
// (MPEG) or full (JPEG) range; anything unknown falls back to BT.709 limited,
// which is what the converted assets use.
struct ColourConversion {
    float matrix[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float offset[3] = {0, 0, 0};

    void build(AVColorSpace space, AVColorRange range) {
        float kr = 0.2126f, kb = 0.0722f;  // BT.709
        switch (space) {
            case AVCOL_SPC_BT470BG:
            case AVCOL_SPC_SMPTE170M:
            case AVCOL_SPC_SMPTE240M:
                kr = 0.299f; kb = 0.114f; break;      // BT.601
            case AVCOL_SPC_BT2020_NCL:
            case AVCOL_SPC_BT2020_CL:
                kr = 0.2627f; kb = 0.0593f; break;    // BT.2020
            default: break;                            // BT.709 / unspecified
        }
        const bool limited = (range != AVCOL_RANGE_JPEG);
        const float kg = 1.0f - kr - kb;
        const float yScale = limited ? 255.0f / 219.0f : 1.0f;
        const float cScale = limited ? 255.0f / 224.0f : 1.0f;
        const float yOff = limited ? 16.0f / 255.0f : 0.0f;
        const float cOff = 0.5f;

        matrix[0][0] = yScale;
        matrix[0][1] = 0.0f;
        matrix[0][2] = 2.0f * (1.0f - kr) * cScale;
        matrix[1][0] = yScale;
        matrix[1][1] = -(2.0f * kb * (1.0f - kb) / kg) * cScale;
        matrix[1][2] = -(2.0f * kr * (1.0f - kr) / kg) * cScale;
        matrix[2][0] = yScale;
        matrix[2][1] = 2.0f * (1.0f - kb) * cScale;
        matrix[2][2] = 0.0f;
        for (int r = 0; r < 3; ++r) {
            offset[r] = -matrix[r][0] * yOff - (matrix[r][1] + matrix[r][2]) * cOff;
        }
    }
};

constexpr int kSlots = 2;  // staging buffers per video: decoder thread <-> GPU

struct Video {
    // identity
    std::string path;
    uint32_t slot = 0;   // layer of the video texture array

    // published state, guarded by `mutex`
    VideoTextureService::VideoState st;

    // decoded geometry
    int width = 0, height = 0;
    int chromaWidth = 0, chromaHeight = 0;
    int64_t packedBytes = 0;
    double frameDuration = 1.0 / 30.0;
    VideoConvertPushConstants pc{};

    // decoder (worker thread only)
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    SwsContext* sws = nullptr;      // only for non 4:2:0 8-bit software formats
    int streamIndex = -1;
    AVRational timeBase{1, 1};
    bool hwDecode = false;
    bool eof = false;
    bool flushSent = false;
    int64_t fallbackPts = 0;        // used when the container omits timestamps
    AVFrame* frame = nullptr;
    AVFrame* swFrame = nullptr;
    AVPacket* pkt = nullptr;

    // worker <-> main thread
    std::thread worker;
    std::mutex mutex;
    std::condition_variable requestCv;  // main -> worker
    std::condition_variable slotCv;     // main -> worker (staging slot freed)
    bool quit = false;
    bool started = false;
    bool hasRequest = false;
    uint64_t requestSeq = 0;
    uint64_t servedSeq = 0;
    int64_t requestPts = 0;
    bool forceSeek = false;
    bool hasPublished = false;
    int publishedSlot = -1;
    int64_t publishedPts = 0;
    int64_t materializedPts = INT64_MIN;
    bool uploadInFlight = false;
    int slotCursor = 0;
    bool slotInUse[kSlots] = {false, false};

    // GPU
    Buffer slots[kSlots];
    VkImage yuvImage = VK_NULL_HANDLE;
    VmaAllocation yuvAlloc = VK_NULL_HANDLE;
    VkImageView yuvView = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkImageLayout yuvLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint64_t yuvBytes = 0;
    uint64_t stagingBytes = 0;

    // instrumentation accumulators, guarded by mutex
    double decodeMsSum = 0.0;
    uint64_t decodeMsCount = 0;
    double seekMsSum = 0.0;
    uint64_t seekMsCount = 0;

    ~Video() {
        if (worker.joinable()) worker.join();
        if (pkt) av_packet_free(&pkt);
        if (frame) av_frame_free(&frame);
        if (swFrame) av_frame_free(&swFrame);
        if (sws) sws_freeContext(sws);
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
    }
};

}  // namespace

struct VideoTextureService::Impl : public std::enable_shared_from_this<VideoTextureService::Impl> {
    VulkanApp* app = nullptr;
    TextureArrayManager* arrays = nullptr;
    bool initialized = false;
    bool alive = true;  // false after shutdown(): deferred callbacks become no-ops

    std::vector<std::unique_ptr<Video>> videos;

    // compute pipeline, shared by every video
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    // Dedicated video texture array: slotCount RGBA8 layers with a full mip
    // chain, sampled by the video layer renderer. Each clip owns one layer and
    // materials reference it through textureIndex; the material texture arrays
    // are never touched by the video path.
    uint32_t slotCount = 0;
    static constexpr uint32_t kMaxSlots = 32;
    TextureImage videoArray;
    VkSampler videoSampler = VK_NULL_HANDLE;
    // Derived material maps for the same slots: RG = tangent-space normal XY,
    // B = bump/height, A = roughness. One mip level (normal/height data must
    // not be averaged; the video shader samples LOD 0 explicitly).
    TextureImage derivedArray;
    std::array<VkImageView, kMaxSlots> derivedStorageViews{};
    std::array<std::array<VkImageView, 3>, kMaxSlots> derivedPreviewViews{};
    std::array<std::array<ImTextureID, 3>, kMaxSlots> derivedPreviews{};
    std::array<VkImageLayout, kMaxSlots> derivedLayouts{};
    std::array<VkImageView, kMaxSlots> slotStorageViews{};
    std::array<VkImageView, kMaxSlots> slotPreviewViews{};
    std::array<ImTextureID, kMaxSlots> slotPreviews{};
    std::array<VkImageLayout, kMaxSlots> slotLayouts{};
    uint32_t videoWidth = 0;
    uint32_t videoHeight = 0;
    uint32_t videoMips = 1;
    // VULKAN_VIDEO_HASH=1: read mip 0 and mip 2 of slot 0 back and hash them.

    std::vector<std::unique_ptr<Video>> retired;  // unloaded clips freed at shutdown

    // Backend probe: one cached hardware device per usable backend, best first.
    // Creating a device is not proof that a codec can decode with it (Vulkan
    // Video advertises AV_PIX_FMT_VULKAN even when VK_KHR_video_decode_queue is
    // missing, and libavcodec then falls back to software silently), so the
    // backend is confirmed per clip from the negotiated pixel format.
    std::vector<std::pair<AVHWDeviceType, AVBufferRef*>> hwCandidates;
    std::string backendName = "software";

    // configuration (env overrides)
    std::string backendPreference = "auto";
    bool debugLog = false;
    bool benchPending = false;
    double diagTimer = 0.0;

    ~Impl() {
        for (auto& candidate : hwCandidates) {
            if (candidate.second) av_buffer_unref(&candidate.second);
        }
        hwCandidates.clear();
    }

    // ── small helpers ─────────────────────────────────────────────────────
    int64_t secondsToPts(const Video& v, double seconds) const {
        const double tick = (v.timeBase.num > 0 && v.timeBase.den > 0) ? av_q2d(v.timeBase) : (1.0 / 30.0);
        return static_cast<int64_t>(std::llround(seconds / std::max(1e-9, tick)));
    }

    double ptsToSeconds(const Video& v, int64_t pts) const {
        const double tick = (v.timeBase.num > 0 && v.timeBase.den > 0) ? av_q2d(v.timeBase) : (1.0 / 30.0);
        return static_cast<double>(pts) * tick;
    }

    int64_t frameTicks(const Video& v) const {
        const double tick = (v.timeBase.num > 0 && v.timeBase.den > 0) ? av_q2d(v.timeBase) : (1.0 / 30.0);
        return std::max<int64_t>(1, static_cast<int64_t>(std::llround(v.frameDuration / std::max(1e-9, tick))));
    }

    bool requestSuperseded(Video& v, uint64_t seq) {
        std::lock_guard<std::mutex> lk(v.mutex);
        return !alive || v.quit || v.requestSeq != seq;
    }

    void log(const std::string& msg) const { std::cout << "[VideoTexture] " << msg << std::endl; }
    void warn(const std::string& msg) const { std::cerr << "[VideoTexture] " << msg << std::endl; }

    // ── backend probe: Vulkan Video, then VAAPI, then software ────────────
    static const char* backendLabel(AVHWDeviceType type) {
        switch (type) {
            case AV_HWDEVICE_TYPE_VULKAN: return "vulkan-video";
            case AV_HWDEVICE_TYPE_VAAPI: return "vaapi";
            default: return "software";
        }
    }

    static std::string firstRenderNode() {
        std::error_code ec;
        for (int i = 128; i < 136; ++i) {
            const std::string candidate = "/dev/dri/renderD" + std::to_string(i);
            if (std::filesystem::exists(candidate, ec)) return candidate;
        }
        return {};
    }

    // Creates (and caches) one hardware device context. Whether a codec can
    // actually decode with it is decided later, per clip, in openDecoder().
    bool addCandidate(AVHWDeviceType type, const char* device, std::string* error) {
        for (auto& candidate : hwCandidates) {
            if (candidate.first == type) return true;
        }
        AVBufferRef* ref = nullptr;
        const int rc = av_hwdevice_ctx_create(&ref, type, device, nullptr, 0);
        if (rc < 0) {
            if (error) {
                char err[AV_ERROR_MAX_STRING_SIZE] = {0};
                av_strerror(rc, err, sizeof(err));
                *error = err;
            }
            if (ref) av_buffer_unref(&ref);
            return false;
        }
        hwCandidates.emplace_back(type, ref);
        return true;
    }

    void probeBackend() {
        auto add = [&](AVHWDeviceType type) -> bool {
            std::string node;
            if (type == AV_HWDEVICE_TYPE_VAAPI) {
                node = firstRenderNode();
                if (node.empty()) {
                    log("hardware decode backend 'vaapi' unavailable: no /dev/dri/renderD* node");
                    return false;
                }
            }
            std::string error;
            if (!addCandidate(type, node.empty() ? nullptr : node.c_str(), &error)) {
                log(std::string("hardware decode backend '") + backendLabel(type) + "' unavailable: " + error);
                return false;
            }
            return true;
        };

        if (backendPreference == "software") {
            log("decode backend forced to software (VULKAN_VIDEO_BACKEND=software)");
        } else {
            const bool autoSelect = backendPreference.empty() || backendPreference == "auto";
            if (autoSelect || backendPreference == "vulkan") add(AV_HWDEVICE_TYPE_VULKAN);
            if (autoSelect || backendPreference == "vaapi") add(AV_HWDEVICE_TYPE_VAAPI);
        }

        if (hwCandidates.empty()) {
            backendName = "software";
            log("decode backend: software");
        } else {
            std::string list;
            for (const auto& candidate : hwCandidates) {
                if (!list.empty()) list += ", ";
                list += backendLabel(candidate.first);
            }
            backendName = backendLabel(hwCandidates.front().first);
            log("decode backend candidates: " + list + " (confirmed per clip, software fallback)");
        }
    }

    bool codecHwConfig(const AVCodec* codec, AVHWDeviceType type, const AVCodecHWConfig** out) const {
        for (int i = 0;; ++i) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
            if (!config) return false;
            if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && config->device_type == type) {
                if (out) *out = config;
                return true;
            }
        }
    }

    // Opens a decoder and confirms with the first decoded frame that the
    // hardware path really engages. libavcodec reports the requested hardware
    // pixel format right after avcodec_open2 even when the hwaccel cannot be
    // initialised (e.g. Vulkan Video without VK_KHR_video_decode_queue); it
    // then falls back to software on the first frame. The frame format is the
    // first reliable proof. The stream is rewound afterwards.
    bool probeHardwareFrame(Video& v, const AVCodecHWConfig* config) {
        AVFrame* frame = av_frame_alloc();
        AVPacket* pkt = av_packet_alloc();
        if (!frame || !pkt) {
            av_frame_free(&frame);
            av_packet_free(&pkt);
            return false;
        }
        bool hardware = false;
        while (!hardware) {
            const int readRc = av_read_frame(v.fmt, pkt);
            if (readRc < 0) break;
            if (pkt->stream_index != v.streamIndex) {
                av_packet_unref(pkt);
                continue;
            }
            const bool sent = avcodec_send_packet(v.dec, pkt) >= 0;
            av_packet_unref(pkt);
            if (!sent) break;
            const int rc = avcodec_receive_frame(v.dec, frame);
            if (rc == 0) {
                hardware = (frame->format == config->pix_fmt);
                av_frame_unref(frame);
                break;
            }
            if (rc != AVERROR(EAGAIN)) break;
        }
        avcodec_flush_buffers(v.dec);
        av_seek_frame(v.fmt, v.streamIndex, 0, AVSEEK_FLAG_BACKWARD);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        return hardware;
    }

    // ── video array + slot priming ────────────────────────────────────────
    // The video array is dedicated to clips: slotCount RGBA8 layers, base level
    // only, sampled by the video layer renderer. Materials of kind Video
    // reference a slot through textureIndex; the material texture arrays are
    // untouched.
    bool createVideoArray(uint32_t width, uint32_t height) {
        videoWidth = std::max(1u, width);
        videoHeight = std::max(1u, height);
        // Base level only: the frame is uploaded every tick, and sampling the
        // level just written avoids both stale-mip driver paths and the
        // per-frame blit/layout chain across array layers.
        videoMips = 1;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {videoWidth, videoHeight, 1};
        imageInfo.mipLevels = videoMips;
        imageInfo.arrayLayers = slotCount;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        app->createImageWithVma(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, videoArray.image,
                                videoArray.allocation, memory, "VideoTextureService: video array");
        if (videoArray.image == VK_NULL_HANDLE) return false;
        videoArray.mipLevels = videoMips;

        VkImageViewCreateInfo arrayView{};
        arrayView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        arrayView.image = videoArray.image;
        arrayView.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrayView.format = VK_FORMAT_R8G8B8A8_UNORM;
        arrayView.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, videoMips, 0, slotCount};
        if (vkCreateImageView(app->getDevice(), &arrayView, nullptr, &videoArray.view) != VK_SUCCESS) return false;
        app->resources.addImageView(videoArray.view, "VideoTextureService: video array view");

        // Derived normal/bump/roughness array (same slots, base level only).
        VkImageCreateInfo derivedInfo = imageInfo;
        derivedInfo.mipLevels = 1;
        derivedInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        app->createImageWithVma(derivedInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, derivedArray.image,
                                derivedArray.allocation, memory, "VideoTextureService: derived map array");
        if (derivedArray.image == VK_NULL_HANDLE) return false;
        derivedArray.mipLevels = 1;

        VkImageViewCreateInfo derivedView = arrayView;
        derivedView.image = derivedArray.image;
        derivedView.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, slotCount};
        if (vkCreateImageView(app->getDevice(), &derivedView, nullptr, &derivedArray.view) != VK_SUCCESS) return false;
        app->resources.addImageView(derivedArray.view, "VideoTextureService: derived map array view");

        for (uint32_t i = 0; i < slotCount; ++i) {
            VkImageViewCreateInfo slotView{};
            slotView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            slotView.image = videoArray.image;
            slotView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            slotView.format = VK_FORMAT_R8G8B8A8_UNORM;
            slotView.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1};
            if (vkCreateImageView(app->getDevice(), &slotView, nullptr, &slotStorageViews[i]) != VK_SUCCESS) return false;
            app->resources.addImageView(slotStorageViews[i], "VideoTextureService: slot storage view");

            VkImageViewCreateInfo derivedSlotView = slotView;
            derivedSlotView.image = derivedArray.image;
            if (vkCreateImageView(app->getDevice(), &derivedSlotView, nullptr, &derivedStorageViews[i]) != VK_SUCCESS)
                return false;
            app->resources.addImageView(derivedStorageViews[i], "VideoTextureService: derived slot storage view");

            slotLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
            derivedLayouts[i] = VK_IMAGE_LAYOUT_UNDEFINED;
            primeSlot(i);
        }

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        // The array has one level (the frame just uploaded); nearest mip mode
        // is explicit so no driver can select a stale generated level.
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // Wrap: video materials project UVs from world position (triplanar /
        // SDF-spherical), so coordinates outside [0,1] must tile the clip.
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.maxLod = 0.0f;
        if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &videoSampler) != VK_SUCCESS) {
            warn("failed to create the video array sampler");
            return false;
        }
        app->resources.addSampler(videoSampler, "VideoTextureService: video sampler");
        return true;
    }

    // Clear one slot to opaque black and leave it sampleable. A
    // material pointing at an empty/loading slot therefore samples a defined
    // color that contributes nothing to the layer blend.
    void primeSlot(uint32_t slot) {
        if (videoArray.image == VK_NULL_HANDLE || slot >= slotCount) return;
        app->runSingleTimeCommands([&](VkCommandBuffer cmd) {
            app->recordTransitionImageLayoutLayer(cmd, videoArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  slotLayouts[slot], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  videoMips, slot, 1);
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = videoMips;
            range.baseArrayLayer = slot;
            range.layerCount = 1;
            VkClearColorValue value{};
            value.float32[0] = 0.0f;
            value.float32[1] = 0.0f;
            value.float32[2] = 0.0f;
            value.float32[3] = 1.0f;
            vkCmdClearColorImage(cmd, videoArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
            app->recordTransitionImageLayoutLayer(cmd, videoArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, videoMips, slot, 1);

            // Neutral derived maps: flat normal (0.5, 0.5), mid bump and
            // roughness, so an unloaded slot contributes no fake relief.
            app->recordTransitionImageLayoutLayer(cmd, derivedArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  derivedLayouts[slot], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  1, slot, 1);
            VkImageSubresourceRange derivedRange{};
            derivedRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            derivedRange.baseMipLevel = 0;
            derivedRange.levelCount = 1;
            derivedRange.baseArrayLayer = slot;
            derivedRange.layerCount = 1;
            VkClearColorValue derivedValue{};
            derivedValue.float32[0] = 0.5f;
            derivedValue.float32[1] = 0.5f;
            derivedValue.float32[2] = 0.5f;
            derivedValue.float32[3] = 0.5f;
            vkCmdClearColorImage(cmd, derivedArray.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &derivedValue,
                                 1, &derivedRange);
            app->recordTransitionImageLayoutLayer(cmd, derivedArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, slot, 1);
        });
        slotLayouts[slot] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        derivedLayouts[slot] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // ── open / decode (worker thread, plus startup benchmark while idle) ──
    bool openDecoder(Video& v) {
        if (avformat_open_input(&v.fmt, v.path.c_str(), nullptr, nullptr) < 0) {
            warn("cannot open " + v.path);
            return false;
        }
        if (avformat_find_stream_info(v.fmt, nullptr) < 0) {
            warn("no stream info in " + v.path);
            return false;
        }
        const AVCodec* codec = nullptr;
        v.streamIndex = av_find_best_stream(v.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        if (v.streamIndex < 0 || !codec) {
            warn("no video stream in " + v.path);
            return false;
        }
        AVStream* stream = v.fmt->streams[v.streamIndex];
        v.timeBase = stream->time_base;

        v.dec = avcodec_alloc_context3(codec);
        if (!v.dec) return false;
        if (avcodec_parameters_to_context(v.dec, stream->codecpar) < 0) return false;

        // Hardware decode: try the cached devices in preference order and keep
        // the first that is actually used. libavcodec opens a decoder even when
        // the hardware format cannot be initialised (it silently negotiates a
        // software format instead), so the negotiated pixel format is the only
        // reliable indication of the backend that will decode the clip.
        AVHWDeviceType chosenType = AV_HWDEVICE_TYPE_NONE;
        std::vector<AVHWDeviceType> unusable;
        for (const auto& candidate : hwCandidates) {
            const AVCodecHWConfig* config = nullptr;
            if (!codecHwConfig(codec, candidate.first, &config)) continue;
            avcodec_free_context(&v.dec);
            v.dec = avcodec_alloc_context3(codec);
            if (!v.dec) return false;
            if (avcodec_parameters_to_context(v.dec, stream->codecpar) < 0) return false;
            v.dec->pix_fmt = config->pix_fmt;
            v.dec->hw_device_ctx = av_buffer_ref(candidate.second);
            if (!v.dec->hw_device_ctx) continue;
            const bool openedOk = avcodec_open2(v.dec, codec, nullptr) >= 0;
            const bool negotiatedHardware = openedOk && v.dec->hw_device_ctx != nullptr &&
                                            v.dec->pix_fmt == config->pix_fmt;
            const bool usingHardware = negotiatedHardware && probeHardwareFrame(v, config);
            if (usingHardware) {
                chosenType = candidate.first;
                break;
            }
            std::string reason;
            if (negotiatedHardware) {
                reason = "advertised but fell back to software for " +
                         std::string(avcodec_get_name(codec->id)) + ", trying the next backend";
            } else {
                reason = "cannot decode " + std::string(avcodec_get_name(codec->id)) +
                         " on this device, trying the next backend";
            }
            warn(std::string("hardware decode backend '") + backendLabel(candidate.first) + "' " + reason);
            unusable.push_back(candidate.first);
        }
        if (chosenType == AV_HWDEVICE_TYPE_NONE) {
            avcodec_free_context(&v.dec);
            v.dec = avcodec_alloc_context3(codec);
            if (!v.dec) return false;
            if (avcodec_parameters_to_context(v.dec, stream->codecpar) < 0) return false;
            if (avcodec_open2(v.dec, codec, nullptr) < 0) {
                warn("cannot open decoder for " + v.path);
                return false;
            }
        }
        // Drop candidates that failed for this codec: later clips would only
        // repeat an initialisation that cannot succeed (and log the same
        // failure again).
        for (AVHWDeviceType type : unusable) {
            for (size_t i = 0; i < hwCandidates.size(); ++i) {
                if (hwCandidates[i].first != type) continue;
                if (hwCandidates[i].second) av_buffer_unref(&hwCandidates[i].second);
                hwCandidates.erase(hwCandidates.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        v.hwDecode = chosenType != AV_HWDEVICE_TYPE_NONE;
        v.st.backend = v.hwDecode ? backendLabel(chosenType) : "software";
        v.st.codec = avcodec_get_name(codec->id);

        const AVRational guessed = av_guess_frame_rate(v.fmt, stream, nullptr);
        const double fps = (guessed.num > 0 && guessed.den > 0) ? av_q2d(guessed) : 0.0;
        v.st.fps = fps;
        v.frameDuration = (fps > 0.01) ? 1.0 / fps : (1.0 / 30.0);
        if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
            v.st.duration = stream->duration * av_q2d(stream->time_base);
        } else if (v.fmt->duration != AV_NOPTS_VALUE && v.fmt->duration > 0) {
            v.st.duration = static_cast<double>(v.fmt->duration) / AV_TIME_BASE;
        }
        v.st.frameCount = (stream->nb_frames > 0)
                              ? static_cast<int64_t>(stream->nb_frames)
                              : static_cast<int64_t>(std::max(0.0, v.st.duration) / std::max(0.01, v.frameDuration));
        v.st.compressedBytes = fileBytes(v.path);

        int64_t keyframes = 0;
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(60, 15, 100)
        // FFmpeg 6.1+: the seek index is behind accessors instead of AVStream fields
        const int entries = avformat_index_get_entries_count(stream);
        for (int i = 0; i < entries; ++i) {
            const AVIndexEntry* entry = avformat_index_get_entry(stream, i);
            if (entry && (entry->flags & AVINDEX_KEYFRAME)) ++keyframes;
        }
#else
        for (unsigned i = 0; i < stream->nb_index_entries; ++i) {
            if (stream->index_entries[i].flags & AVINDEX_KEYFRAME) ++keyframes;
        }
#endif
        v.st.keyframeCount = keyframes;

        const AVCodecParameters* par = stream->codecpar;
        if (par->width <= 0 || par->height <= 0 || (par->width % 2) || (par->height % 2)) {
            warn("unsupported frame geometry (4:2:0 needs even dimensions) in " + v.path);
            return false;
        }
        v.width = par->width;
        v.height = par->height;
        v.chromaWidth = v.width / 2;
        v.chromaHeight = v.height / 2;
        v.packedBytes = static_cast<int64_t>(v.width) * (v.height + v.chromaHeight);
        v.st.width = v.width;
        v.st.height = v.height;

        ColourConversion colour;
        colour.build(par->color_space, par->color_range);
        v.pc.lumaWidth = v.width;
        v.pc.lumaHeight = v.height;
        v.pc.chromaWidth = v.chromaWidth;
        v.pc.chromaHeight = v.chromaHeight;
        v.pc.dstWidth = static_cast<int32_t>(videoWidth);
        v.pc.dstHeight = static_cast<int32_t>(videoHeight);
        for (int r = 0; r < 3; ++r) {
            v.pc.offset[r] = colour.offset[r];
            v.pc.matrixRow0[r] = colour.matrix[0][r];
            v.pc.matrixRow1[r] = colour.matrix[1][r];
            v.pc.matrixRow2[r] = colour.matrix[2][r];
        }

        v.frame = av_frame_alloc();
        v.swFrame = av_frame_alloc();
        v.pkt = av_packet_alloc();
        return v.frame && v.swFrame && v.pkt;
    }

    // Bring the decoded frame into CPU memory as 8-bit 4:2:0 (nv12 or yuv420p).
    bool downloadFrame(Video& v) {
        AVFrame* src = v.frame;
        const bool isPlanar420 = src->format == AV_PIX_FMT_YUV420P || src->format == AV_PIX_FMT_NV12;
        if (v.hwDecode && !isPlanar420) {
            av_frame_unref(v.swFrame);
            if (av_hwframe_transfer_data(v.swFrame, src, 0) < 0) {
                warn("hardware frame download failed on " + v.path);
                return false;
            }
            v.swFrame->pts = src->best_effort_timestamp;
            return true;
        }
        if (!isPlanar420) {
            // rare software formats (yuvj420p, 10-bit, ...): convert to yuv420p
            if (!v.sws) {
                v.sws = sws_getContext(src->width, src->height, static_cast<AVPixelFormat>(src->format),
                                       src->width, src->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!v.sws) return false;
            }
            av_frame_unref(v.swFrame);
            v.swFrame->format = AV_PIX_FMT_YUV420P;
            v.swFrame->width = src->width;
            v.swFrame->height = src->height;
            if (av_frame_get_buffer(v.swFrame, 32) < 0) return false;
            sws_scale(v.sws, src->data, src->linesize, 0, src->height, v.swFrame->data, v.swFrame->linesize);
            v.swFrame->pts = src->best_effort_timestamp;
            return true;
        }
        if (src != v.swFrame) {
            av_frame_unref(v.swFrame);
            if (av_frame_ref(v.swFrame, src) < 0) return false;
            v.swFrame->pts = src->best_effort_timestamp;
        }
        return true;
    }

    // Pack a 4:2:0 frame into the layout the compute shader expects: luma rows,
    // then rows of interleaved U/V pairs (NV12 byte layout). Planar frames are
    // interleaved here; NV12 frames copy row-wise.
    bool packFrame(const Video& v, const AVFrame* f, uint8_t* dst) const {
        const int w = v.width;
        const int h = v.height;
        const int cw = v.chromaWidth;
        const int ch = v.chromaHeight;
        if (f->linesize[0] < 0) return false;
        for (int y = 0; y < h; ++y) {
            std::memcpy(dst + static_cast<size_t>(y) * w, f->data[0] + static_cast<size_t>(y) * f->linesize[0], w);
        }
        uint8_t* chroma = dst + static_cast<size_t>(h) * w;
        if (f->format == AV_PIX_FMT_NV12) {
            if (f->linesize[1] < 0) return false;
            for (int y = 0; y < ch; ++y) {
                std::memcpy(chroma + static_cast<size_t>(y) * w, f->data[1] + static_cast<size_t>(y) * f->linesize[1], w);
            }
        } else {
            if (f->linesize[1] < 0 || f->linesize[2] < 0) return false;
            for (int y = 0; y < ch; ++y) {
                const uint8_t* u = f->data[1] + static_cast<size_t>(y) * f->linesize[1];
                const uint8_t* vv = f->data[2] + static_cast<size_t>(y) * f->linesize[2];
                uint8_t* out = chroma + static_cast<size_t>(y) * w;
                for (int x = 0; x < cw; ++x) {
                    out[2 * x] = u[x];
                    out[2 * x + 1] = vv[x];
                }
            }
        }
        return true;
    }

    // Decode forward until a frame at or after `target` is available; it lands
    // in v.swFrame. A superseding request does not throw the work away: the
    // newest target is adopted and decoding continues forward. That lets the
    // main thread keep the material on the current clock (it may supersede
    // every frame) without the worker restarting a seek, which is exactly what
    // stalled the loop wrap before. Returns false on EOF/error, when the newest
    // target moved backwards (the caller then seeks), or when the target
    // outruns the decoder.
    bool decodeUntil(Video& v, int64_t& target, uint64_t& seq, bool checkSupersede, bool& superseded, uint64_t* decodedCount = nullptr) {
        superseded = false;
        // Bound a single chase to this many seconds of video: a very high
        // playback rate must not trap the worker in an endless decode.
        const int64_t maxChase = std::max<int64_t>(1, static_cast<int64_t>(std::llround(2.0 / std::max(1e-9, v.frameDuration))));
        int64_t chaseFrames = 0;
        while (true) {
            if (checkSupersede && requestSuperseded(v, seq)) {
                int64_t newest = 0;
                {
                    std::lock_guard<std::mutex> g(v.mutex);
                    if (!alive || v.quit) {
                        superseded = true;
                        return false;
                    }
                    newest = v.requestPts;
                    seq = v.requestSeq;
                    v.servedSeq = seq;
                }
                if (newest < target) {
                    superseded = true;  // backwards: the caller has to seek
                    return false;
                }
                target = newest;
            }
            if (v.flushSent && v.eof) return false;
            if (chaseFrames > maxChase) {
                superseded = true;  // target outran the decoder; re-evaluate/seek
                return false;
            }

            int ret = avcodec_receive_frame(v.dec, v.frame);
            if (ret == 0) {
                ++chaseFrames;
                int64_t pts = v.frame->best_effort_timestamp;
                if (pts == AV_NOPTS_VALUE) {
                    v.fallbackPts += frameTicks(v);
                    pts = v.fallbackPts;
                } else {
                    v.fallbackPts = pts;
                }
                if (pts >= target) {
                    if (!downloadFrame(v)) {
                        av_frame_unref(v.frame);
                        return false;
                    }
                    v.swFrame->pts = pts;
                    if (decodedCount) ++(*decodedCount);
                    return true;
                }
                av_frame_unref(v.frame);
                continue;
            }
            if (ret == AVERROR(EAGAIN)) {
                const int read = av_read_frame(v.fmt, v.pkt);
                if (read < 0) {
                    if (!v.flushSent) {
                        avcodec_send_packet(v.dec, nullptr);
                        v.flushSent = true;
                        continue;
                    }
                    v.eof = true;
                    return false;
                }
                if (v.pkt->stream_index == v.streamIndex) {
                    const int sent = avcodec_send_packet(v.dec, v.pkt);
                    av_packet_unref(v.pkt);
                    if (sent < 0 && sent != AVERROR(EAGAIN)) return false;
                } else {
                    av_packet_unref(v.pkt);
                }
                continue;
            }
            if (ret == AVERROR_EOF) {
                v.eof = true;
                return false;
            }
            char err[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, err, sizeof(err));
            warn("decode error on " + v.path + ": " + err);
            return false;
        }
    }

    bool seekTo(Video& v, int64_t target) {
        if (av_seek_frame(v.fmt, v.streamIndex, target, AVSEEK_FLAG_BACKWARD) < 0) {
            warn("seek failed on " + v.path);
            return false;
        }
        avcodec_flush_buffers(v.dec);
        v.eof = false;
        v.flushSent = false;
        return true;
    }

    bool acquireSlot(Video& v, int& outSlot, uint64_t seq) {
        std::unique_lock<std::mutex> lk(v.mutex);
        while (alive && !v.quit) {
            for (int i = 0; i < kSlots; ++i) {
                const int idx = (v.slotCursor + i) % kSlots;
                if (!v.slotInUse[idx]) {
                    v.slotInUse[idx] = true;
                    v.slotCursor = (idx + 1) % kSlots;
                    outSlot = idx;
                    return true;
                }
            }
            if (v.requestSeq != seq) return false;  // superseded while waiting
            v.slotCv.wait_for(lk, std::chrono::milliseconds(5));
        }
        return false;
    }

    void workerLoop(Video* vp) {
        Video& v = *vp;
        std::unique_lock<std::mutex> lk(v.mutex);
        int64_t positionPts = INT64_MIN;  // pts of the last decoded frame
        std::vector<uint8_t> scratch;
        while (alive && !v.quit) {
            v.requestCv.wait(lk, [&] { return !alive || v.quit || (v.hasRequest && v.requestSeq != v.servedSeq); });
            if (!alive || v.quit) break;
            uint64_t seq = v.requestSeq;
            int64_t target = v.requestPts;
            const bool forced = v.forceSeek;
            v.forceSeek = false;
            v.servedSeq = seq;
            v.st.decodePending = true;
            const double requestStart = nowMs();
            lk.unlock();

            bool superseded = false;
            bool published = false;
            // A forward target within this time budget is reached by decoding
            // forward (sequential decode is ~6 ms/frame); a real seek costs
            // ~90 ms, so seeking for a handful of frames is far more expensive
            // than decoding them. The clock advances while a request is being
            // served, so a tight budget made the worker seek on essentially
            // every request, which is what dragged playback behind real time.
            const int64_t jump = std::max<int64_t>(frameTicks(v), static_cast<int64_t>(std::llround(0.5 / std::max(1e-9, av_q2d(v.timeBase)))));
            const bool needSeek = forced || positionPts == INT64_MIN || target < positionPts || (target - positionPts) > jump;

            if (!needSeek) {
                // the frame the material already samples may still cover this request
                std::lock_guard<std::mutex> g(v.mutex);
                if (v.materializedPts != INT64_MIN && v.materializedPts >= target &&
                    (v.materializedPts - target) <= std::max<int64_t>(1, frameTicks(v))) {
                    ++v.st.cacheHits;
                    ++v.st.requestsServed;
                    v.st.decodePending = v.hasRequest && v.requestSeq != v.servedSeq;
                    lk.lock();
                    continue;
                }
            }

            if (needSeek) {
                if (!seekTo(v, target)) {
                    std::lock_guard<std::mutex> g(v.mutex);
                    ++v.st.requestsServed;
                    v.st.decodePending = v.hasRequest && v.requestSeq != v.servedSeq;
                    lk.lock();
                    continue;
                }
                // The decoder now sits on the keyframe at or before `target`, so
                // decoding forward reaches it. Recording that position keeps a
                // superseding request with a forward target from seeking again
                // (re-seeking on every supersession is what stalled the loop).
                positionPts = target;
            }

            uint64_t decoded = 0;
            if (decodeUntil(v, target, seq, true, superseded, &decoded)) {
                positionPts = v.swFrame->pts;
                int slot = -1;
                if (acquireSlot(v, slot, seq)) {
                    if (scratch.size() < static_cast<size_t>(v.packedBytes)) scratch.resize(static_cast<size_t>(v.packedBytes));
                    if (packFrame(v, v.swFrame, scratch.data()) && v.slots[slot].mappedData) {
                        std::memcpy(v.slots[slot].mappedData, scratch.data(), static_cast<size_t>(v.packedBytes));
                        std::lock_guard<std::mutex> g(v.mutex);
                        v.hasPublished = true;
                        v.publishedSlot = slot;
                        v.publishedPts = v.swFrame->pts;
                        published = true;
                    } else {
                        std::lock_guard<std::mutex> g(v.mutex);
                        v.slotInUse[slot] = false;
                        v.slotCv.notify_all();
                    }
                } else {
                    std::lock_guard<std::mutex> g(v.mutex);
                    ++v.st.requestsSuperseded;
                }
                av_frame_unref(v.swFrame);
            } else if (superseded) {
                std::lock_guard<std::mutex> g(v.mutex);
                ++v.st.requestsSuperseded;
            }

            {
                std::lock_guard<std::mutex> g(v.mutex);
                v.st.framesDecoded += decoded;
                ++v.st.requestsServed;
                if (published) {
                    const double ms = nowMs() - requestStart;
                    if (needSeek) {
                        v.seekMsSum += ms;
                        ++v.seekMsCount;
                    } else {
                        v.decodeMsSum += ms;
                        ++v.decodeMsCount;
                    }
                }
                v.st.decodePending = v.hasRequest && v.requestSeq != v.servedSeq;
            }
            lk.lock();
        }
        lk.unlock();
    }

    void startWorker(Video& v) {
        if (v.started || !alive) return;
        v.started = true;
        {
            std::lock_guard<std::mutex> lk(v.mutex);
            v.requestPts = 0;
            ++v.requestSeq;
            v.hasRequest = true;
            v.forceSeek = true;
        }
        v.worker = std::thread([this, &v]() { workerLoop(&v); });
    }

    // ── GPU resources ─────────────────────────────────────────────────────
    bool createPipeline() {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(app->getDevice(), &layoutInfo, nullptr, &descLayout) != VK_SUCCESS) {
            warn("failed to create compute descriptor set layout");
            return false;
        }
        app->resources.addDescriptorSetLayout(descLayout, "VideoTextureService: descLayout");

        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.offset = 0;
        range.size = sizeof(VideoConvertPushConstants);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            warn("failed to create compute pipeline layout");
            return false;
        }
        app->resources.addPipelineLayout(pipelineLayout, "VideoTextureService: pipelineLayout");

        VkShaderModule module = app->getOrCreateShaderModule("shaders/computation/video/VideoYuvToRgba.comp.spv");
        if (module == VK_NULL_HANDLE) {
            warn("compute shader shaders/computation/video/VideoYuvToRgba.comp.spv missing (run `make shaders`)");
            return false;
        }
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(app->getDevice(), app->getPipelineCache(), 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            warn("failed to create compute pipeline");
            return false;
        }
        app->resources.addPipeline(pipeline, "VideoTextureService: pipeline");

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxLod = 1.0f;
        if (vkCreateSampler(app->getDevice(), &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
            warn("failed to create sampler");
            return false;
        }
        app->resources.addSampler(sampler, "VideoTextureService: sampler");

        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 64;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 128;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        // The per-video descriptor sets are only (re)written while the device is
        // idle (registration, array rebuild), so no update-after-bind features
        // are needed here.
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 64;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        if (vkCreateDescriptorPool(app->getDevice(), &poolInfo, nullptr, &descPool) != VK_SUCCESS) {
            warn("failed to create descriptor pool");
            return false;
        }
        app->resources.addDescriptorPool(descPool, "VideoTextureService: descriptorPool");
        return true;
    }

    bool writeDescriptors(Video& v) {
        if (v.descSet == VK_NULL_HANDLE || v.yuvView == VK_NULL_HANDLE) return false;
        if (v.slot >= slotCount || slotStorageViews[v.slot] == VK_NULL_HANDLE) return false;
        const VkImageView target = slotStorageViews[v.slot];
        if (target == VK_NULL_HANDLE) return false;

        VkDescriptorImageInfo yuvInfo{};
        yuvInfo.sampler = sampler;
        yuvInfo.imageView = v.yuvView;
        yuvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo targetInfo{};
        targetInfo.imageView = target;
        targetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo derivedInfo{};
        derivedInfo.imageView = derivedStorageViews[v.slot];
        derivedInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[3] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = v.descSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &yuvInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = v.descSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &targetInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = v.descSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &derivedInfo;
        vkUpdateDescriptorSets(app->getDevice(), 3, writes, 0, nullptr);
        return true;
    }

    bool createVideoResources(Video& v, uint32_t slot) {
        v.slot = slot;
        v.st.slot = slot;

        // Packed YUV image: one R8 image, luma rows then interleaved chroma rows.
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8_UNORM;
        imageInfo.extent = {static_cast<uint32_t>(v.width), static_cast<uint32_t>(v.height + v.chromaHeight), 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        app->createImageWithVma(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, v.yuvImage, v.yuvAlloc, memory,
                                "VideoTextureService: packed YUV frame");
        if (v.yuvImage == VK_NULL_HANDLE) return false;
        v.yuvBytes = static_cast<uint64_t>(v.width) * static_cast<uint64_t>(v.height + v.chromaHeight);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = v.yuvImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(app->getDevice(), &viewInfo, nullptr, &v.yuvView) != VK_SUCCESS) return false;
        app->resources.addImageView(v.yuvView, "VideoTextureService: yuv view");

        // Host-visible staging slots (persistently mapped, reused every frame).
        v.stagingBytes = 0;
        for (int i = 0; i < kSlots; ++i) {
            v.slots[i] = app->createBuffer(static_cast<VkDeviceSize>(v.packedBytes),
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                           false);
            if (v.slots[i].buffer == VK_NULL_HANDLE) return false;
            v.stagingBytes += static_cast<uint64_t>(v.packedBytes);
        }
        v.st.gpuYuvBytes = v.yuvBytes;
        v.st.stagingBytes = v.stagingBytes;

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descLayout;
        if (vkAllocateDescriptorSets(app->getDevice(), &allocInfo, &v.descSet) != VK_SUCCESS) return false;
        app->resources.addDescriptorSet(v.descSet, "VideoTextureService: per-video set");
        return writeDescriptors(v);
    }

    // Record and submit one frame: staging -> packed YUV image -> compute
    // conversion into the video's albedo layer (+ mip chain), on the graphics
    // queue. Two orderings make this safe without ever blocking the CPU:
    //  * the submit waits on the frame timeline semaphore at its current value,
    //    so the write cannot race a frame that still samples the previous
    //    contents (every frame submitted so far has finished executing);
    //  * the completion semaphore is registered with the app, so the following
    //    frames wait for the new contents before sampling them.
    void submitUpload(Video& v, int slot, int64_t pts, size_t index) {
        const uint32_t slotIndex = v.slot;
        const uint32_t mips = std::max(1u, videoMips);
        VideoConvertPushConstants pc = v.pc;
        pc.dstWidth = static_cast<int32_t>(videoWidth);
        pc.dstHeight = static_cast<int32_t>(videoHeight);
        const int32_t dstW = pc.dstWidth;
        const int32_t dstH = pc.dstHeight;

        const double t0 = nowMs();
        VkCommandBuffer cmd = app->allocatePrimaryCommandBuffer();
        if (cmd == VK_NULL_HANDLE) {
            warn("no command buffer for a video upload");
            std::lock_guard<std::mutex> lk(v.mutex);
            v.slotInUse[slot] = false;
            v.uploadInFlight = false;
            v.slotCv.notify_all();
            return;
        }
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            warn("vkBeginCommandBuffer failed for a video upload");
            std::lock_guard<std::mutex> lk(v.mutex);
            v.slotInUse[slot] = false;
            v.uploadInFlight = false;
            v.slotCv.notify_all();
            return;
        }

        // packed YUV image: staging -> device local
        app->recordTransitionImageLayoutLayer(cmd, v.yuvImage, VK_FORMAT_R8_UNORM, v.yuvLayout,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {static_cast<uint32_t>(v.width), static_cast<uint32_t>(v.height + v.chromaHeight), 1};
        vkCmdCopyBufferToImage(cmd, v.slots[slot].buffer, v.yuvImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        v.yuvLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        app->recordTransitionImageLayoutLayer(cmd, v.yuvImage, VK_FORMAT_R8_UNORM,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);

        // video array slot: SHADER_READ_ONLY -> GENERAL, dispatch, mip chain,
        // back (the same sequence TextureMixer uses for generated layers).
        app->recordTransitionImageLayoutLayer(cmd, videoArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                              slotLayouts[slotIndex], VK_IMAGE_LAYOUT_GENERAL, mips, slotIndex, 1);
        app->recordTransitionImageLayoutLayer(cmd, derivedArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                              derivedLayouts[slotIndex], VK_IMAGE_LAYOUT_GENERAL, 1, slotIndex, 1);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &v.descSet, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (static_cast<uint32_t>(dstW) + 15) / 16, (static_cast<uint32_t>(dstH) + 15) / 16, 1);
        app->recordTransitionImageLayoutLayer(cmd, derivedArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                              1, slotIndex, 1);
        app->recordTransitionImageLayoutLayer(cmd, videoArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips, slotIndex, 1);
        // Single base level: finish in SHADER_READ_ONLY. No mip blits, no
        // per-level layout changes for the video array.
        app->recordTransitionImageLayoutLayer(cmd, videoArray.image, VK_FORMAT_R8G8B8A8_UNORM,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mips, slotIndex, 1);
        slotLayouts[slotIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        derivedLayouts[slotIndex] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::vector<VkSemaphore> waits;
        std::vector<uint64_t> waitValues;
        const VkSemaphore timeline = app->getFrameTimelineSemaphore();
        const uint64_t timelineValue = app->getFrameTimelineValue();
        if (timeline != VK_NULL_HANDLE) {
            waits.push_back(timeline);
            waitValues.push_back(timelineValue);
        }
        VkSemaphore completion = VK_NULL_HANDLE;
        VkFence fence = app->submitCommandBufferAsyncToQueue(cmd, app->getGraphicsQueue(), &completion, waits, true, {},
                                                             waitValues);
        if (fence == VK_NULL_HANDLE) {
            warn("video upload submit failed for " + v.path);
            std::lock_guard<std::mutex> lk(v.mutex);
            v.slotInUse[slot] = false;
            v.uploadInFlight = false;
            v.slotCv.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(v.mutex);
            v.materializedPts = pts;
            v.st.frameTime = ptsToSeconds(v, pts);
            v.st.frameReady = true;
            v.st.lastUploadMs = nowMs() - t0;
            ++v.st.framesUploaded;
        }
        // The staging slot and the layer layout are released once the GPU has
        // consumed the upload (deferred callbacks run before the fence retires).
        std::weak_ptr<Impl> weak = weak_from_this();
        app->deferDestroyUntilFence(fence, [weak, index, slot]() {
            if (auto impl = weak.lock()) {
                if (index >= impl->videos.size()) return;
                Video& vv = *impl->videos[index];
                if (vv.slot < impl->slotCount) impl->slotLayouts[vv.slot] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                {
                    std::lock_guard<std::mutex> lk(vv.mutex);
                    vv.slotInUse[slot] = false;
                    vv.uploadInFlight = false;
                }
                vv.slotCv.notify_all();
            }
        });
    }

    void updateVideo(Video& v, size_t index, float dt) {
        bool submitSlot = false;
        int slot = -1;
        int64_t pts = 0;
        {
            std::lock_guard<std::mutex> lk(v.mutex);
            if (v.st.playing) {
                v.st.clock += static_cast<double>(dt) * v.st.rate;
                const double duration = v.st.duration;
                if (duration > 0.0) {
                    if (v.st.clock >= duration) {
                        if (v.st.looping) {
                            v.st.clock = std::fmod(v.st.clock, duration);
                            v.forceSeek = true;
                        } else {
                            v.st.clock = duration;
                            v.st.playing = false;
                        }
                    } else if (v.st.clock < 0.0) {
                        if (v.st.looping) {
                            v.st.clock = std::fmod(v.st.clock, duration) + duration;
                            v.forceSeek = true;
                        } else {
                            v.st.clock = 0.0;
                            v.st.playing = false;
                        }
                    }
                }
            }

            const int64_t desired = secondsToPts(v, v.st.clock);
            const int64_t halfFrame = std::max<int64_t>(1, frameTicks(v) / 2);
            // forceSeek marks a loop wrap/seek: request it even when the
            // materialized timestamp happens to be close to the new clock, so
            // playback restarts immediately at the loop point.
            const bool stale = v.forceSeek || (v.materializedPts == INT64_MIN) ||
                               std::llabs(desired - v.materializedPts) > halfFrame;
            // A clock-driven request supersedes whatever the worker is decoding
            // (the worker adopts the newest target instead of restarting, see
            // decodeUntil), so the material always tracks the current clock
            // rather than a stale request target. Issuing it while the previous
            // frame is still uploading lets the next decode overlap that upload
            // instead of waiting a whole upload round-trip for it; the worker
            // has a second staging slot for exactly this. Only one publish may
            // be queued at a time (!hasPublished) so a slot is never orphaned.
            if (stale && !v.hasPublished) {
                v.requestPts = desired;
                ++v.requestSeq;
                v.hasRequest = true;
                v.st.decodePending = true;
                v.requestCv.notify_one();
            }
            if (v.hasPublished && !v.uploadInFlight) {
                v.hasPublished = false;
                v.uploadInFlight = true;
                submitSlot = true;
                slot = v.publishedSlot;
                pts = v.publishedPts;
            }
        }
        if (submitSlot && slot >= 0) submitUpload(v, slot, pts, index);
    }
};

#else  // !HAVE_FFMPEG

struct VideoTextureService::Impl {
    bool alive = false;
};

#endif  // HAVE_FFMPEG

// ── public API ────────────────────────────────────────────────────────────
VideoTextureService::VideoTextureService() : impl_(std::make_shared<Impl>()) {}

VideoTextureService::~VideoTextureService() { shutdown(); }

bool VideoTextureService::backendCompiledIn() {
#ifdef HAVE_FFMPEG
    return true;
#else
    return false;
#endif
}

bool VideoTextureService::isVideoFile(const std::string& path) {
    const std::string ext = lowerExtension(path);
    return ext == ".mp4" || ext == ".mov" || ext == ".mkv" || ext == ".webm" || ext == ".m4v" || ext == ".avi";
}

std::vector<std::string> VideoTextureService::listVideoFiles(const std::string& directory) {
    std::vector<std::string> files;
    std::error_code ec;
    if (directory.empty() || !std::filesystem::is_directory(directory, ec)) return files;
    for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file()) continue;
        const std::string path = entry.path().string();
        if (isVideoFile(path)) files.push_back(path);
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string VideoTextureService::findAssetDirectory() {
    // The working directory is bin/: videos/<category> holds the clips staged
    // by the build (converted by scripts/convert_videos.sh), while
    // ../videos/<category> holds the converted clips committed in the
    // repository. Both contain converted clips only; raw masters live in
    // videos/raw/ and are never scanned. Staged folders are checked first.
    static const char* candidates[] = {
        "videos/base", "videos/extra", "../videos/base", "../videos/extra",
    };
    std::error_code ec;
    for (const char* candidate : candidates) {
        if (std::filesystem::is_directory(candidate, ec)) {
            if (!listVideoFiles(candidate).empty()) return candidate;
        }
    }
    return std::string();
}

std::vector<VideoTextureService::AvailableVideo> VideoTextureService::listAvailableVideos() {
    std::vector<AvailableVideo> result;

    // The working directory is bin/: videos/<category> there holds the clips
    // staged by the build (converted by scripts/convert_videos.sh), while
    // ../videos/<category> holds the converted clips committed in the
    // repository. Scanning the staged folder first (and deduplicating by name)
    // prefers the copy next to the executable; both folders carry the
    // base/extra category directly from their name. Raw masters in
    // videos/raw/ are intentionally never scanned.
    std::unordered_set<std::string> seen;
    auto scan = [&](const char* dir, const char* category) {
        for (const std::string& file : listVideoFiles(dir)) {
            AvailableVideo clip;
            clip.name = std::filesystem::path(file).stem().string();
            if (!seen.insert(clip.name).second) continue;
            clip.category = category;
            clip.path = file;
            result.push_back(std::move(clip));
        }
    };
    scan("videos/base", "base");
    scan("../videos/base", "base");
    scan("videos/extra", "extra");
    scan("../videos/extra", "extra");

    // base first, then name
    std::sort(result.begin(), result.end(), [](const AvailableVideo& a, const AvailableVideo& b) {
        if (a.category != b.category) return a.category == "base";
        return a.name < b.name;
    });
    return result;
}

#ifdef HAVE_FFMPEG

bool VideoTextureService::initialize(VulkanApp* app, TextureArrayManager* arrays, uint32_t slotCount) {
    Impl& impl = *impl_;
    if (!app || !arrays) {
        std::cerr << "[VideoTextureService] initialize needs a VulkanApp and a TextureArrayManager\n";
        return false;
    }
    if (impl.initialized) return true;
    impl.app = app;
    impl.arrays = arrays;
    if (const char* slots = std::getenv("VULKAN_VIDEO_SLOTS")) {
        const int requested = std::atoi(slots);
        if (requested > 0) slotCount = static_cast<uint32_t>(requested);
    }
    impl.slotCount = std::clamp(slotCount, 1u, Impl::kMaxSlots);
    impl.backendPreference = envString("VULKAN_VIDEO_BACKEND");
    if (impl.backendPreference.empty()) impl.backendPreference = "auto";
    impl.debugLog = envFlag("VULKAN_VIDEO_DEBUG");
    impl.benchPending = envFlag("VULKAN_VIDEO_BENCH");
    impl.probeBackend();
    std::cout << "[VideoTextureService] ffmpeg " << av_version_info() << std::endl;

    if (!impl.createPipeline()) return false;
    // The video array is dedicated to clips and sized like the material arrays
    // (same texel density as a normal material layer), with slotCount layers.
    if (!impl.createVideoArray(arrays->width, arrays->height)) {
        std::cerr << "[VideoTextureService] failed to create the video texture array\n";
        return false;
    }
    std::cout << "[VideoTextureService] video array: " << impl.slotCount << " slots at "
              << impl.videoWidth << "^2 (base level, "
              << (static_cast<uint64_t>(impl.videoWidth) * impl.videoHeight * 4ull * impl.slotCount / (1024 * 1024))
              << " MiB total)" << std::endl;
    impl.initialized = true;
    return true;
}

int VideoTextureService::loadVideo(const std::string& path, const std::string& name) {
    Impl& impl = *impl_;
    if (!impl.initialized) {
        std::cerr << "[VideoTextureService] loadVideo before initialize\n";
        return -1;
    }
    // First free slot (clips are rarely unloaded, so a linear scan is fine).
    uint32_t slot = impl.slotCount;
    for (uint32_t candidate = 0; candidate < impl.slotCount; ++candidate) {
        bool used = false;
        for (const auto& video : impl.videos) {
            if (video && video->slot == candidate) { used = true; break; }
        }
        if (!used) { slot = candidate; break; }
    }
    if (slot >= impl.slotCount) {
        std::cerr << "[VideoTextureService] video array full (" << impl.slotCount
                  << " slots, VULKAN_VIDEO_SLOTS); cannot load " << path << "\n";
        return -1;
    }

    auto video = std::make_unique<Video>();
    video->path = path;
    video->st.name = name.empty() ? std::filesystem::path(path).stem().string() : name;
    video->st.path = path;
    // Clips always start playing and looping. Pausing is an explicit action in
    // the Video Textures widget (Play/Pause controls); no startup switch can
    // leave a material on a frozen frame.
    video->st.playing = true;
    video->st.looping = true;
    video->st.rate = 1.0f;

    if (!impl.openDecoder(*video)) return -1;
    impl.primeSlot(slot);
    if (!impl.createVideoResources(*video, slot)) {
        std::cerr << "[VideoTextureService] GPU resources failed for " << path << "\n";
        return -1;
    }

    std::cout << "[VideoTexture] +" << video->st.name << " -> video slot " << slot
              << " (" << video->st.width << "x" << video->st.height << ", " << video->st.codec << ", "
              << video->st.backend << ", " << video->st.duration << "s @ " << video->st.fps << " fps, "
              << video->st.keyframeCount << " keyframes, " << (video->st.compressedBytes / (1024 * 1024))
              << " MiB compressed)\n";

    impl.videos.push_back(std::move(video));
    if (!impl.benchPending) impl.startWorker(*impl.videos.back());
    return static_cast<int>(slot);
}

void VideoTextureService::unloadVideo(uint32_t slot) {
    Impl& impl = *impl_;
    for (size_t i = 0; i < impl.videos.size(); ++i) {
        if (!impl.videos[i] || impl.videos[i]->slot != slot) continue;
        std::unique_ptr<Video> video = std::move(impl.videos[i]);
        impl.videos.erase(impl.videos.begin() + static_cast<std::ptrdiff_t>(i));
        {
            std::lock_guard<std::mutex> lk(video->mutex);
            video->quit = true;
        }
        video->requestCv.notify_all();
        video->slotCv.notify_all();
        if (video->worker.joinable()) video->worker.join();
        impl.primeSlot(slot);
        // Free the per-clip GPU objects with the rest at shutdown (they are
        // registered with the app's resource manager); keep them alive until
        // then so no in-flight deferred callback can observe a dangling slot.
        impl.retired.push_back(std::move(video));
        std::cout << "[VideoTexture] -slot " << slot << " unloaded\n";
        return;
    }
}

void VideoTextureService::shutdown() {
    Impl& impl = *impl_;
    impl.alive = false;
    for (auto& video : impl.videos) {
        if (!video) continue;
        {
            std::lock_guard<std::mutex> lk(video->mutex);
            video->quit = true;
        }
        video->requestCv.notify_all();
        video->slotCv.notify_all();
        if (video->worker.joinable()) video->worker.join();
    }
    impl.videos.clear();
    impl.retired.clear();
    impl.videoArray = TextureImage{};
    impl.derivedArray = TextureImage{};
    impl.videoSampler = VK_NULL_HANDLE;
    impl.slotStorageViews.fill(VK_NULL_HANDLE);
    impl.slotPreviewViews.fill(VK_NULL_HANDLE);
    impl.slotPreviews.fill(0);
    impl.derivedStorageViews.fill(VK_NULL_HANDLE);
    for (auto& views : impl.derivedPreviewViews) views.fill(VK_NULL_HANDLE);
    for (auto& previews : impl.derivedPreviews) previews.fill(0);
    impl.initialized = false;
}

size_t VideoTextureService::count() const { return impl_->videos.size(); }

bool VideoTextureService::empty() const { return impl_->videos.empty(); }

uint32_t VideoTextureService::slotCapacity() const { return impl_->slotCount; }

bool VideoTextureService::isSlotLoaded(uint32_t slot) const {
    for (const auto& video : impl_->videos) {
        if (video && video->slot == slot) return true;
    }
    return false;
}

VkImage VideoTextureService::videoArrayImage() const { return impl_->videoArray.image; }
VkImageView VideoTextureService::videoArrayView() const { return impl_->videoArray.view; }
VkSampler VideoTextureService::videoArraySampler() const { return impl_->videoSampler; }
VkImageView VideoTextureService::videoDerivedArrayView() const { return impl_->derivedArray.view; }
VkSampler VideoTextureService::videoDerivedArraySampler() const { return impl_->videoSampler; }
uint32_t VideoTextureService::videoArrayMips() const { return impl_->videoMips; }
uint32_t VideoTextureService::videoArrayWidth() const { return impl_->videoWidth; }
uint32_t VideoTextureService::videoArrayHeight() const { return impl_->videoHeight; }

ImTextureID VideoTextureService::previewTexture(uint32_t slot) {
    Impl& impl = *impl_;
    if (!impl.initialized || slot >= impl.slotCount) return 0;
    if (impl.slotPreviews[slot]) return impl.slotPreviews[slot];
    if (impl.videoArray.image == VK_NULL_HANDLE || impl.videoSampler == VK_NULL_HANDLE) return 0;

    // Separate per-slot 2D view with the full mip chain (the compute-target
    // storage view is mip-0 only); created on first preview use.
    if (impl.slotPreviewViews[slot] == VK_NULL_HANDLE) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = impl.videoArray.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, impl.videoMips, slot, 1};
        if (vkCreateImageView(impl.app->getDevice(), &viewInfo, nullptr,
                              &impl.slotPreviewViews[slot]) != VK_SUCCESS) {
            return 0;
        }
        impl.app->resources.addImageView(impl.slotPreviewViews[slot],
                                         "VideoTextureService: preview view");
    }
    impl.slotPreviews[slot] = (ImTextureID)ImGui_ImplVulkan_AddTexture(
        impl.videoSampler, impl.slotPreviewViews[slot], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return impl.slotPreviews[slot];
}

ImTextureID VideoTextureService::derivedPreviewTexture(uint32_t slot, uint32_t map) {
    Impl& impl = *impl_;
    if (!impl.initialized || slot >= impl.slotCount || map > 2) return 0;
    if (impl.derivedPreviews[slot][map]) return impl.derivedPreviews[slot][map];
    if (impl.derivedArray.image == VK_NULL_HANDLE || impl.videoSampler == VK_NULL_HANDLE) return 0;

    VkImageView& view = impl.derivedPreviewViews[slot][map];
    if (view == VK_NULL_HANDLE) {
        // Swizzle the packed derived map into the channel the preview tab
        // shows: normal = RG + constant blue, bump = B, roughness = A.
        VkComponentMapping components{};
        if (map == 0) {
            components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                          VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_ONE};
        } else if (map == 1) {
            components = {VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_B,
                          VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_ONE};
        } else {
            components = {VK_COMPONENT_SWIZZLE_A, VK_COMPONENT_SWIZZLE_A,
                          VK_COMPONENT_SWIZZLE_A, VK_COMPONENT_SWIZZLE_ONE};
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = impl.derivedArray.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.components = components;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, slot, 1};
        if (vkCreateImageView(impl.app->getDevice(), &viewInfo, nullptr, &view) != VK_SUCCESS) {
            return 0;
        }
        impl.app->resources.addImageView(view, "VideoTextureService: derived preview view");
    }
    impl.derivedPreviews[slot][map] = (ImTextureID)ImGui_ImplVulkan_AddTexture(
        impl.videoSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return impl.derivedPreviews[slot][map];
}

void VideoTextureService::invalidatePreviews() {
    // Only drop the cached handles: the ImGui descriptor pool they came from has
    // already been (or is about to be) recreated, so RemoveTexture is not valid.
    impl_->slotPreviews.fill(0);
    for (auto& previews : impl_->derivedPreviews) previews.fill(0);
}

VideoTextureService::VideoState VideoTextureService::state(size_t index) const {
    Impl& impl = *impl_;
    VideoState out;
    if (index >= impl.videos.size()) return out;
    Video& v = *impl.videos[index];
    std::lock_guard<std::mutex> lk(v.mutex);
    out = v.st;
    const auto average = [](double sum, uint64_t n) { return n ? sum / static_cast<double>(n) : 0.0; };
    out.avgDecodeMs = average(v.decodeMsSum, v.decodeMsCount);
    out.avgSeekMs = average(v.seekMsSum, v.seekMsCount);
    return out;
}

std::vector<VideoTextureService::VideoState> VideoTextureService::states() const {
    std::vector<VideoState> out;
    out.reserve(count());
    for (size_t i = 0; i < count(); ++i) out.push_back(state(i));
    return out;
}

uint32_t VideoTextureService::slotOf(size_t index) const {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return 0;
    return impl.videos[index]->slot;
}

void VideoTextureService::update(float dt) {
    Impl& impl = *impl_;
    if (!impl.initialized || impl.videos.empty()) return;
    if (impl.benchPending) {
        // Benchmark mode: runBenchmark() measures on the idle decoders and then
        // starts the playback workers. If it was never called, start them here.
        impl.diagTimer += dt;
        if (impl.diagTimer > 10.0) {
            impl.benchPending = false;
            for (auto& video : impl.videos) {
                if (video) impl.startWorker(*video);
            }
        }
        return;
    }
    for (size_t i = 0; i < impl.videos.size(); ++i) {
        impl.updateVideo(*impl.videos[i], i, dt);
    }
    impl.diagTimer += dt;
    if (impl.debugLog && impl.diagTimer >= 5.0) {
        impl.diagTimer = 0.0;
        logDiagnostics();
    }
}

void VideoTextureService::onArraysRecreated() {
    Impl& impl = *impl_;
    if (!impl.initialized) return;
    // The video array keeps its own size across material-array rebuilds, so the
    // slots and their views stay valid: only refresh the conversion push
    // constants and rebind the descriptors (defensive; views are unchanged).
    for (auto& video : impl.videos) {
        if (!video) continue;
        video->pc.dstWidth = static_cast<int32_t>(impl.videoWidth);
        video->pc.dstHeight = static_cast<int32_t>(impl.videoHeight);
        impl.writeDescriptors(*video);
    }
}

void VideoTextureService::setPlaying(size_t index, bool playing) {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return;
    Video& v = *impl.videos[index];
    std::lock_guard<std::mutex> lk(v.mutex);
    v.st.playing = playing;
    if (playing) v.requestCv.notify_all();
}

void VideoTextureService::togglePlaying(size_t index) {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return;
    bool playing = false;
    {
        Video& v = *impl.videos[index];
        std::lock_guard<std::mutex> lk(v.mutex);
        playing = v.st.playing;
    }
    setPlaying(index, !playing);
}

void VideoTextureService::setPlayingAll(bool playing) {
    for (size_t i = 0; i < count(); ++i) setPlaying(i, playing);
}

void VideoTextureService::setLooping(size_t index, bool looping) {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return;
    Video& v = *impl.videos[index];
    std::lock_guard<std::mutex> lk(v.mutex);
    v.st.looping = looping;
}

void VideoTextureService::setRate(size_t index, float rate) {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return;
    Video& v = *impl.videos[index];
    std::lock_guard<std::mutex> lk(v.mutex);
    v.st.rate = std::clamp(rate, -16.0f, 16.0f);
}

void VideoTextureService::seek(size_t index, double seconds) {
    Impl& impl = *impl_;
    if (index >= impl.videos.size()) return;
    Video& v = *impl.videos[index];
    std::lock_guard<std::mutex> lk(v.mutex);
    v.st.clock = std::max(0.0, seconds);
    if (v.st.duration > 0.0) v.st.clock = std::min(v.st.clock, v.st.duration);
    v.forceSeek = true;
    v.requestPts = impl.secondsToPts(v, v.st.clock);
    ++v.requestSeq;
    v.hasRequest = true;
    v.hasPublished = false;
    v.requestCv.notify_all();
}

void VideoTextureService::seekAll(double seconds) {
    for (size_t i = 0; i < count(); ++i) seek(i, seconds);
}

void VideoTextureService::logDiagnostics() const {
    Impl& impl = *impl_;
    if (impl.videos.empty()) return;
    const uint64_t layerBytes = static_cast<uint64_t>(impl.videoWidth) * impl.videoHeight * 4ull;
    uint64_t compressed = 0, gpu = 0, staging = 0, naive = 0, decoded = 0, uploaded = 0, seeks = 0, hits = 0;
    for (size_t i = 0; i < impl.videos.size(); ++i) {
        const VideoState s = state(i);
        const uint64_t videoNaive = layerBytes * static_cast<uint64_t>(std::max<int64_t>(0, s.frameCount));
        compressed += static_cast<uint64_t>(std::max<int64_t>(0, s.compressedBytes));
        gpu += s.gpuYuvBytes;
        staging += s.stagingBytes;
        decoded += s.framesDecoded;
        uploaded += s.framesUploaded;
        seeks += s.seeks;
        hits += s.cacheHits;
        naive += videoNaive;
        std::cout << "[VideoTexture::diag] " << s.name << " slot=" << s.slot << " backend=" << s.backend
                  << " codec=" << s.codec << " " << s.width << "x" << s.height << " clock=" << s.clock << "/"
                  << s.duration << " playing=" << (s.playing ? 1 : 0) << " rate=" << s.rate
                  << " frameT=" << s.frameTime << " decoded=" << s.framesDecoded
                  << " uploaded=" << s.framesUploaded << " seeks=" << s.seeks << " hits=" << s.cacheHits
                  << " keyframes=" << s.keyframeCount << " avgDecode=" << s.avgDecodeMs
                  << "ms avgSeek=" << s.avgSeekMs << "ms lastUpload=" << s.lastUploadMs
                  << "ms yuv=" << (s.gpuYuvBytes / 1024) << "KiB staging=" << (s.stagingBytes / 1024)
                  << "KiB naiveRgba=" << (videoNaive / (1024 * 1024)) << "MiB" << std::endl;
    }
    const double ratio = naive ? (100.0 * static_cast<double>(gpu + staging) / static_cast<double>(naive)) : 0.0;
    std::cout << "[VideoTexture::diag] total videos=" << impl.videos.size() << " decoded=" << decoded
              << " uploaded=" << uploaded << " seeks=" << seeks << " cacheHits=" << hits
              << " compressed=" << (compressed / (1024 * 1024)) << "MiB"
              << " gpuYuv=" << (gpu / (1024 * 1024)) << "MiB"
              << " staging=" << (staging / (1024 * 1024)) << "MiB"
              << " naiveRgbaAllFrames=" << (naive / (1024 * 1024)) << "MiB (" << ratio
              << "% of a fully decoded copy)" << std::endl;
}

void VideoTextureService::runBenchmark(size_t videosToTest, size_t seeksPerVideo) {
    Impl& impl = *impl_;
    if (!impl.initialized || impl.videos.empty()) {
        impl.benchPending = false;
        return;
    }
    const size_t countToTest = std::min(videosToTest, impl.videos.size());
    std::cout << "[VideoTexture::bench] measuring decode/seek behaviour (backend=" << impl.backendName
              << ", " << countToTest << " video(s), " << seeksPerVideo << " seek(s) each)" << std::endl;
    for (size_t i = 0; i < countToTest; ++i) {
        Video& v = *impl.videos[i];
        bool superseded = false;
        impl.seekTo(v, 0);
        uint64_t decoded = 0;
        const uint64_t frames = static_cast<uint64_t>(std::max(1.0, 1.0 / std::max(0.01, v.frameDuration)));
        const int64_t tick = impl.frameTicks(v);
        const double t0 = nowMs();
        for (uint64_t f = 0; f < frames; ++f) {
            int64_t target = static_cast<int64_t>(f + 1) * tick;
            uint64_t seq = 0;
            if (!impl.decodeUntil(v, target, seq, false, superseded, &decoded)) break;
            av_frame_unref(v.swFrame);
        }
        const double decodeMs = nowMs() - t0;
        const double fps = (decodeMs > 0.0 && decoded > 0) ? (1000.0 * static_cast<double>(decoded) / decodeMs) : 0.0;

        const int64_t span = std::max<int64_t>(1, impl.secondsToPts(v, std::max(1.0, v.st.duration - 1.0)));
        double seekMsSum = 0.0;
        size_t seekCount = 0;
        uint64_t seed = 12345u;
        for (size_t s = 0; s < seeksPerVideo && v.st.duration > 2.0; ++s) {
            seed = seed * 1103515245u + 12345u;
            const int64_t pts = static_cast<int64_t>(seed % static_cast<uint64_t>(span));
            const double start = nowMs();
            if (!impl.seekTo(v, pts)) continue;
            int64_t decodeTarget = pts;
            uint64_t seq = 0;
            if (impl.decodeUntil(v, decodeTarget, seq, false, superseded)) {
                seekMsSum += nowMs() - start;
                ++seekCount;
                av_frame_unref(v.swFrame);
            }
        }

        const uint64_t layerBytes = static_cast<uint64_t>(impl.videoWidth) * impl.videoHeight * 4ull;
        const double layerMiB = static_cast<double>(layerBytes) / (1024.0 * 1024.0);
        const double naiveMiB = layerMiB * static_cast<double>(v.st.frameCount);
        const double streamingMiB = (static_cast<double>(v.st.gpuYuvBytes) + static_cast<double>(v.st.stagingBytes)) / (1024.0 * 1024.0);
        std::cout << "[VideoTexture::bench] " << v.st.name << " sequential decode " << fps << " fps ("
                  << (fps > 0.0 ? 1000.0 / fps : 0.0) << " ms/frame), seek+decode avg="
                  << (seekCount ? seekMsSum / static_cast<double>(seekCount) : 0.0) << " ms over " << seekCount
                  << " seeks; resident per-frame bytes " << streamingMiB << " MiB vs " << naiveMiB
                  << " MiB fully decoded (" << (100.0 * streamingMiB / std::max(0.001, naiveMiB)) << "%)"
                  << std::endl;
    }
    impl.benchPending = false;
    for (auto& video : impl.videos) {
        if (video) impl.startWorker(*video);
    }
    std::cout << "[VideoTexture::bench] done; playback workers started" << std::endl;
}

#else  // !HAVE_FFMPEG

bool VideoTextureService::initialize(VulkanApp*, TextureArrayManager*, uint32_t) {
    std::cout << "[VideoTextureService] built without FFmpeg (libav* not found at configure time); "
                 "video textures are unavailable. Install libavformat/libavcodec/libavutil and rebuild.\n";
    return false;
}

int VideoTextureService::loadVideo(const std::string&, const std::string&) { return -1; }
void VideoTextureService::unloadVideo(uint32_t) {}
bool VideoTextureService::isSlotLoaded(uint32_t) const { return false; }
uint32_t VideoTextureService::slotCapacity() const { return 0; }
VkImage VideoTextureService::videoArrayImage() const { return VK_NULL_HANDLE; }
VkImageView VideoTextureService::videoArrayView() const { return VK_NULL_HANDLE; }
VkSampler VideoTextureService::videoArraySampler() const { return VK_NULL_HANDLE; }
VkImageView VideoTextureService::videoDerivedArrayView() const { return VK_NULL_HANDLE; }
VkSampler VideoTextureService::videoDerivedArraySampler() const { return VK_NULL_HANDLE; }
uint32_t VideoTextureService::videoArrayMips() const { return 1; }
uint32_t VideoTextureService::videoArrayWidth() const { return 0; }
uint32_t VideoTextureService::videoArrayHeight() const { return 0; }
ImTextureID VideoTextureService::previewTexture(uint32_t) { return 0; }
ImTextureID VideoTextureService::derivedPreviewTexture(uint32_t, uint32_t) { return 0; }
void VideoTextureService::invalidatePreviews() {}
void VideoTextureService::shutdown() {}
size_t VideoTextureService::count() const { return 0; }
bool VideoTextureService::empty() const { return true; }
VideoTextureService::VideoState VideoTextureService::state(size_t) const { return {}; }
std::vector<VideoTextureService::VideoState> VideoTextureService::states() const { return {}; }
uint32_t VideoTextureService::slotOf(size_t) const { return 0; }
void VideoTextureService::update(float) {}
void VideoTextureService::onArraysRecreated() {}
void VideoTextureService::setPlaying(size_t, bool) {}
void VideoTextureService::togglePlaying(size_t) {}
void VideoTextureService::setPlayingAll(bool) {}
void VideoTextureService::setLooping(size_t, bool) {}
void VideoTextureService::setRate(size_t, float) {}
void VideoTextureService::seek(size_t, double) {}
void VideoTextureService::seekAll(double) {}
void VideoTextureService::logDiagnostics() const {}
void VideoTextureService::runBenchmark(size_t, size_t) {}

#endif  // HAVE_FFMPEG
