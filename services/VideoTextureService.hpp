#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanApp;
class TextureArrayManager;

// ── Video textures ────────────────────────────────────────────────────────
//
// A video texture is an animated material: a compressed source file is decoded
// on demand for the timestamp its playback clock asks for, converted to RGBA8
// on the GPU (shaders/computation/video/VideoYuvToRgba.comp) and written into
// one layer of the engine's dedicated **video texture array**. Materials of
// kind MaterialKind::Video sample that array through textureIndex (the video
// slot), so several materials can reference the same clip and clips can be
// loaded/unloaded at runtime without touching the material texture arrays.
//
// Video clips are loaded from the Videos widget (or programmatically with
// loadVideo) into a fixed-capacity array of slots; `videos/base/` holds the 3
// clips MyApp loads at setup, `videos/extra/` is the optional local library.
//
// Decoding backend (runtime detected per clip, best first):
//   1. Vulkan Video  - libavcodec's `vulkan` hwaccel, requires
//                      VK_KHR_video_decode_queue + the codec's decode extension.
//   2. VAAPI         - the DRM render node.
//   3. software      - libavcodec's CPU decoders.
// A device can be created and still be unusable for a codec (Vulkan Video
// advertises AV_PIX_FMT_VULKAN even when the decode extension is missing and
// libavcodec silently falls back to software), so the backend is confirmed per
// clip from the negotiated pixel format and reported in VideoState::backend.
//
// Memory model (see logDiagnostics):
//   * compressed: the source file on disk, never fully loaded.
//   * per clip  : one packed YUV R8 image (~1.5 bytes/pixel) + two host-visible
//                 staging buffers of the same size.
//   * shared    : the video array (slotCount × RGBA8 + mips), sampled by the
//                 video layer renderer.
// A "decode everything" implementation would hold width*height*4 bytes for
// every frame of every clip; logDiagnostics prints that comparison.
//
// Playback is timestamp driven: the clock advances with the frame delta times
// the playback rate, seeks map to the preceding keyframe plus a bounded decode
// run, and the render loop never blocks on the decoder - until a new frame is
// ready the material keeps sampling the previous one.
//
// Environment switches:
//   VULKAN_NO_VIDEO=1              do not create the video service/array
//   VULKAN_VIDEO_SLOTS=<n>         video array capacity (default 8)
//   VULKAN_VIDEO_BACKEND=auto|vulkan|vaapi|software
//   VULKAN_VIDEO_AUTOPLAY=<n|all>  clips that start playing (default 4);
//                                  paused clips still materialize their frame
//   VULKAN_VIDEO_DEBUG=1           diagnostics every 5 s
//   VULKAN_VIDEO_BENCH=1           decode/seek benchmark at startup
class VideoTextureService {
public:
    static constexpr uint32_t kDefaultSlotCount = 8;

    // Runtime state of one clip; a snapshot for the widget and diagnostics.
    struct VideoState {
        std::string name;
        std::string path;
        std::string backend;        // "vulkan-video" | "vaapi" | "software"
        std::string codec;          // "h264", "hevc", ...
        int width = 0;              // decoded (luma) size
        int height = 0;
        double duration = 0.0;      // seconds
        double fps = 0.0;           // guessed frame rate
        int64_t frameCount = 0;     // from the container index when known
        int64_t keyframeCount = 0;
        int64_t compressedBytes = 0;
        uint32_t slot = 0;          // video-array layer (material textureIndex)

        // Playback
        bool playing = false;
        bool looping = true;
        float rate = 1.0f;
        double clock = 0.0;         // playback position, seconds
        double frameTime = 0.0;     // timestamp of the frame the material sees
        bool frameReady = false;
        bool decodePending = false;

        // Instrumentation
        uint64_t framesDecoded = 0;
        uint64_t framesUploaded = 0;
        uint64_t seeks = 0;
        uint64_t requestsServed = 0;
        uint64_t requestsSuperseded = 0;
        uint64_t cacheHits = 0;      // request satisfied without decoding
        double avgDecodeMs = 0.0;    // per request served by decoding forward
        double avgSeekMs = 0.0;      // per request served by seeking first
        double lastUploadMs = 0.0;   // CPU time to record one frame upload
        uint64_t gpuYuvBytes = 0;    // packed YUV image
        uint64_t stagingBytes = 0;   // host-visible staging slots
    };

    // One clip discovered on disk. `category` is "base" (comes from
    // videos/base/, loaded by MyApp at setup) or "extra" (videos/extra/, local
    // library). `path` is the converted asset when one exists, else the source.
    struct AvailableVideo {
        std::string path;
        std::string name;
        std::string category;
    };

    VideoTextureService();
    ~VideoTextureService();
    VideoTextureService(const VideoTextureService&) = delete;
    VideoTextureService& operator=(const VideoTextureService&) = delete;

    // True when the build includes the FFmpeg backend (HAVE_FFMPEG).
    static bool backendCompiledIn();

    // Whether name ends in a supported video container extension.
    static bool isVideoFile(const std::string& path);

    // Sorted list of video files in `directory` (empty when missing).
    static std::vector<std::string> listVideoFiles(const std::string& directory);

    // Locate a video asset directory: the converted clips mirror the source
    // folders in videos/<base|extra>/converted/; sources are checked as a
    // fallback. Returns "" when nothing is found.
    static std::string findAssetDirectory();

    // Discover loadable clips from videos/base and videos/extra (preferring the
    // converted videos/<category>/converted/<name> when present). Also lists
    // converted-only clips (e.g. a bin/ copy without sources).
    static std::vector<AvailableVideo> listAvailableVideos();

    // Create the compute pipeline, the video texture array (slotCount layers,
    // same size as the material arrays) and the per-slot resources. Returns
    // false when video textures are unavailable (no FFmpeg, no device).
    bool initialize(VulkanApp* app, TextureArrayManager* arrays,
                    uint32_t slotCount = kDefaultSlotCount);

    // Stop the workers and release the video-array GPU objects. Safe to call
    // twice; also called from the destructor.
    void shutdown();

    // Load a clip into the first free video-array slot and start its decoder.
    // Returns the slot index (>= 0) or -1 when the clip cannot be opened or the
    // array is full. The slot is primed to black first and then driven by the
    // decoded frames. `name` defaults to the file name.
    int loadVideo(const std::string& path, const std::string& name = std::string());

    // Stop the clip in `slot`, clear the slot to black and free it. Safe on an
    // empty slot.
    void unloadVideo(uint32_t slot);

    bool isSlotLoaded(uint32_t slot) const;
    uint32_t slotCapacity() const;

    // Video array accessors for the video layer renderer (sampler2DArray).
    VkImage videoArrayImage() const;
    VkImageView videoArrayView() const;
    VkSampler videoArraySampler() const;
    uint32_t videoArrayMips() const;
    uint32_t videoArrayWidth() const;
    uint32_t videoArrayHeight() const;

    size_t count() const;
    bool empty() const;

    // Snapshot of one loaded clip (thread-safe).
    VideoState state(size_t index) const;
    std::vector<VideoState> states() const;

    uint32_t slotOf(size_t index) const;

    // Per-frame pump: advances the clocks, issues/cancels decode requests,
    // submits finished frames. Never blocks on decoding.
    void update(float dt);

    // The material arrays were recreated (resolution change): the video array
    // keeps its own size, so this only refreshes the conversion push constants
    // and re-primes slots whose target layout was invalidated.
    void onArraysRecreated();

    // ── playback control ──
    void setPlaying(size_t index, bool playing);
    void togglePlaying(size_t index);
    void setPlayingAll(bool playing);
    void setLooping(size_t index, bool looping);
    void setRate(size_t index, float rate);
    void seek(size_t index, double seconds);
    void seekAll(double seconds);

    // ── diagnostics ──
    // One line per clip plus a total: backend, decode/seek latency, cache,
    // GPU bytes and the comparison against a fully decoded copy.
    void logDiagnostics() const;
    // Measure decode throughput and seek latency per clip and print the
    // memory/throughput comparison against the naive "keep every RGBA frame"
    // strategy. Called at startup when VULKAN_VIDEO_BENCH=1 is set.
    void runBenchmark(size_t videos = 2, size_t seeksPerVideo = 6);

private:
    struct Impl;
    // shared_ptr so deferred GPU callbacks can hold a weak reference: a callback
    // that fires after shutdown() (or after the service was destroyed) must not
    // touch the video list.
    std::shared_ptr<Impl> impl_;
};
