// Standard library includes first
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <thread>
#include <filesystem>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <array>
#include <unordered_set>
#include <shared_mutex>
#include <sstream>
#include <cstdint>
#include <zlib.h>
#include <sys/resource.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "vulkan/ubo/UniformObject.hpp"
#include "vulkan/ubo/SkyUniform.hpp"
#include "vulkan/core/VulkanApp.hpp"
#include "vulkan/streaming/PublishTarget.hpp"
#include "vulkan/renderer/SceneRenderer.hpp"
#include "vulkan/renderer/SceneDescriptorLayout.hpp"
#include "vulkan/renderer/SceneQueues.hpp"
#include "vulkan/renderer/RendererUtils.hpp"
#include "utils/LocalScene.hpp"
#include "utils/SvgBrush.hpp"
#include "sdf/SvgPolygonDistanceFunction.hpp"
#include "sdf/SvgShape.hpp"
#include "utils/BlenderExport.hpp"
#include "utils/SceneMaterialTable.hpp"
#include "utils/RemoteScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/SkySettings.hpp"
#include "widgets/CloudSettings.hpp"
#include "widgets/CloudWidget.hpp"
#include "widgets/PhysicsWidget.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/GraphicsQualityWidget.hpp"
#include "widgets/RenderTargetsWidget.hpp"
#include "widgets/ImpostorWidget.hpp"
#include "services/ImpostorService.hpp"
#include "widgets/TextureMixerWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "services/VideoTextureService.hpp"
#include "widgets/VideoTextureWidget.hpp"
#include "widgets/CameraWidget.hpp"
#include "events/ControllerManager.hpp"
#include "widgets/ControllerParametersWidget.hpp"
#include "widgets/GamepadWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/VulkanResourcesManagerWidget.hpp"
#include "widgets/QueueTimelineWidget.hpp"
#include "widgets/WindWidget.hpp"
#include "widgets/RaymarchWidget.hpp"
#include "widgets/GrassWidget.hpp"
#include "sdf/types/SdfScene.hpp"
#include "widgets/OctreeExplorerWidget.hpp"
#include "widgets/Brush3dWidget.hpp"
#include "widgets/LayersWidget.hpp"
#include "widgets/MusicWidget.hpp"
#include "widgets/components/FilePicker.hpp"
#include "sdf/AddSignedDistanceOperation.hpp"
#include "sdf/DeleteSignedDistanceOperation.hpp"
#include "sdf/PaintSignedDistanceOperation.hpp"
#include "sdf/SweepSignedDistanceFunction.hpp"
#include "utils/MainSceneLoader.hpp"
#include "space/UniqueChangeCollector.hpp"
#include "space/OctreeFile.hpp"
#include "types/Settings.hpp"
#include "types/WaterSettings.hpp"
#include "utils/GraphicsSettingsCommand.hpp"
#include "widgets/WidgetManager.hpp"
#include "widgets/RadialMenu.hpp"
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/GamepadPublisher.hpp"
#include "events/NunchukPublisher.hpp"
#include "events/MousePublisher.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "events/RebuildBrushEvent.hpp"
#include "events/ApplyBrushToSceneEvent.hpp"
#include "events/SetBrushTextureEvent.hpp"
#include "events/SetBrushControlEvent.hpp"
#include "events/SetBrushPaintModeEvent.hpp"
#include "events/SetBrushDragModeEvent.hpp"
#include "events/SetBrushHSVEvent.hpp"
#include "events/SetBrushSdfTypeEvent.hpp"
#include "events/SetLightEvent.hpp"
#include "events/SetPageEvent.hpp"
#include "events/SetGraphicsQualityEvent.hpp"
#include "events/RadialMenuHandler.hpp"
#include "vulkan/resources/TextureArrayManager.hpp"
#include "vulkan/resources/MaterialManager.hpp"
#include "world/World.hpp"
#include "vulkan/renderer/DescriptorWriter.hpp"
#include "services/TextureMixer.hpp"
#include "services/OctreeTopologyCompute.hpp"
#include "services/ParticlePhysicsSim.hpp"
#include "server/SceneServer.hpp"
#include "vulkan/renderer/water/DropletRenderer.hpp"
#include "services/OctreeTopologyTypes.hpp"
#include "utils/ShadowParams.hpp"
#include "space/ThreadPool.hpp"
#include "space/Octree.hpp"

// Build the {onAdded, onDeleted} renderer lambdas for one main-scene space
// (solid or water). Returns the pair of renderer-side lambdas; the caller
// wires them behind a UniqueChangeCollector dedup stage and dispatches on the
// main thread.
std::pair<Octree::OctreeNodeDataHandler, Octree::OctreeNodeDataHandler> build(SceneRenderer* renderer, VulkanApp* app, Scene* scene,
              Layer layer, float minSize, ThreadPool* genPool, const PublishTarget& target) {

    Octree::OctreeNodeDataHandler onAdded = [renderer, app, scene, layer, minSize, genPool, target](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        ChunkManager::ChunkId cid = static_cast<ChunkManager::ChunkId>(nid);

        // SDF debug cubes are collected inside SceneRenderer::processNodeLayer
        // via scene.requestSDFCubes (mirrors the solid mesh walk) — no separate
        // marker pass needed here.
        // Phase 1: mark dirty and begin build IMMEDIATELY when the octree
        // change is detected (before tessellation is dispatched to the
        // worker pool). This transitions Clean → Queed → BuildingCPU.
        if (renderer->world()) {
            renderer->world()->chunkManager().markDirty(cid, nd.node->version);
            renderer->world()->chunkManager().beginBuild(cid);
        }

        OctreeNodeData nodeCopy = nd;
        // Single-mesh handler: each chunk emits exactly one LoDMesh and it is
        // queued as its own entry — the consumer publishes it into the chunk's
        // stable slot.
        renderer->processNodeLayer(*scene, layer, nid, nodeCopy,
            [renderer, cid, nodeCopy, target](Layer layer_, NodeID nid_, const LoDMesh& lodMesh) {
                // Every published chunk is GPU-resident. There is no host-side
                // vertex mirror; an invalidation without a live slot is dropped.
                if (lodMesh.gpuSlot == UINT32_MAX) {
                    return;
                }
                // lodMesh.lod is the 0-based band level (chunkLod - 1); the
                // stored chunkLod is 1-based. finishBuild only when the mesh is
                // the added node's own rung.
                if (nodeCopy.node->getChunkLod() == lodMesh.lod + 1) {
                    // Phase 3: tessellation complete on a worker thread. The
                    // chunk mesh is complete; only the octree version is
                    // tracked here — GPU data goes through slots.
                    if (renderer->world()) {
                        renderer->world()->chunkManager().finishBuild(cid, lodMesh.version);
                    }
                }
                // Phase 4: Queue for main-thread GPU upload. The map is keyed
                // by the emitting octree node id (one entry per octree node;
                // pushing again for the same node overwrites in place, so the
                // last tessellation result wins).
                std::lock_guard<std::mutex> lock(target.queueMutex);
                target.meshData[nid_] = {layer_, nid_, lodMesh, nodeCopy};
            },
            minSize,
            genPool);
    };

    Octree::OctreeNodeDataHandler onDeleted = [renderer, app, scene, target](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);

        // Forget the node's tessellation dedupe record: the node memory may be
        // reused, and a stale entry would suppress re-tessellation of the new
        // occupant.
        if (auto* ls = dynamic_cast<LocalScene*>(scene)) {
            ls->noteDeletedNode(static_cast<uintptr_t>(nid));
        }

        if (renderer->world()) {
            // One slot per chunk: defer the chunk's single slot until its
            // matching re-publish completes (or it ages out in
            // processPendingMeshes). Don't free immediately — the octree node
            // is reused on edit (same NodeID), so republishing the chunk
            // updates the slot in place and consumes this entry.
            const ChunkManager::ChunkId base = static_cast<ChunkManager::ChunkId>(nid);
            uint32_t sidx = renderer->world()->chunkManager().getSlotIndex(base);
            if (sidx == UINT32_MAX) {
                // Coarse ancestor cells are not tracked by the ChunkManager
                // (only frontier chunks are); resolve their slot through the
                // scene chunk map recorded at publish time.
                std::lock_guard<std::recursive_mutex> lock(target.chunksMutex);
                auto it = target.chunks.find(nid);
                if (it != target.chunks.end()) {
                    sidx = it->second.meshId;
                    target.chunks.erase(it);
                }
            }
            if (sidx != UINT32_MAX) {
                target.deferredSlots[nid] = {sidx, app->getCurrentFrame()};
            }
            renderer->world()->chunkManager().removeChunk(base);
            if (renderer->debugCubeRenderer) renderer->debugCubeRenderer->removeCubeForNode(nid);
            if (renderer->debugSDFRenderer) renderer->debugSDFRenderer->removeCubesForNode(nid);
            if (renderer->sdfRenderer) {
                renderer->sdfRenderer->removeLavaChunk(static_cast<uintptr_t>(nid));
            }
            return;
        }
    };
    return { onAdded, onDeleted };
}

// ── Setup-phase voice alerts ──────────────────────────────────────────────
// Each setup step below announces itself through the shared alert queue in
// sounds/ once it has completed: enqueue synchronously on sounds/play.sh
// (queue order == call order) and leave playback to a detached drainer. This
// is the same queue and script the Makefile alerts use, so app and build
// voices serialize instead of overlapping. No-op when the sounds folder is
// missing (e.g. a standalone copy of bin/).
static void playSetupAlert(const char* name) {
    namespace fs = std::filesystem;
    static const fs::path script = [] {
        std::error_code ec;
        for (const char* candidate : {"../sounds/play.sh", "sounds/play.sh"})
            if (fs::exists(candidate, ec)) return fs::weakly_canonical(candidate, ec);
        return fs::path{};
    }();
    if (script.empty()) return;

    const fs::path sound = script.parent_path() / (std::string(name) + ".mp3");
    std::string cmd = "sh '" + script.string() + "' --enqueue '" + sound.string() + "'";
    cmd += " </dev/null >/dev/null 2>&1; nohup sh '" + script.string()
         + "' --drain </dev/null >/dev/null 2>&1 &";
    const int status = std::system(cmd.c_str());
    (void)status; // Alerts are best-effort; a failed spawn must not affect startup.
}

class MyApp : public VulkanApp, public IEventHandler {
public:
    Settings settings;
    // Last grass mode applied to the renderers; the mode selector only
    // pushes on change so the Grass/Impostors widget toggles are not
    // overridden every frame.
    Settings::GrassMode appliedGrassMode = Settings::GrassMode::None;
    SceneRenderer * sceneRenderer = nullptr;
    // Scene render queues are owned by the generic VulkanApp framework via its
    // SceneQueues member (forward-declared, created and configured below in
    // setup()). MyApp no longer duplicates them; the getXxxQueue() forwarders on
    // VulkanApp delegate to SceneQueues, keeping this class free of scene-staging
    // boilerplate.
    World * world = nullptr;
    // Remote scene (rendering the chunk-streaming server). Set at startup:
    // MyApp tries the WebSocket first and falls back to the local octrees
    // when it is unreachable. Owned here; World::activeScene() decides which
    // scene the renderer consumes.
    std::unique_ptr<RemoteScene> remoteScene;
    bool useRemote = false;
    // "host:port" of the chunk server. Overridable via --remote=HOST:PORT,
    // env VULKAN_REMOTE=HOST:PORT; --local (or VULKAN_REMOTE=off) skips the
    // connection attempt entirely.
    std::string remoteEndpoint = "127.0.0.1:8080";
    bool remoteDisabled = false;
    std::shared_ptr<Brush3dWidget> brush3dWidget;
    // Shared brush entries edited by Brush3dWidget (owned by MyApp)
    Brush3dManager brushManager;
    // Cached sweep start position so applyBrushToScene uses the same pair as the preview
    glm::vec3 cachedSweepStart = glm::vec3(0.0f);

    // ── Brush undo history (per layer) ─────────────────────────────────────
    // The layer's octree is a pure result of its base scene plus the brushes
    // applied to it. We keep one compressed snapshot of that base (taken
    // before the first brush on the layer) and the ordered brush events. Undo
    // restores the base and replays the remaining events, so overlapping
    // add/remove/paint strokes are reproduced exactly instead of trying to
    // invert a non-invertible SDF edit.
    struct BrushHistoryEvent {
        BrushEntry entry;
        glm::vec3 sweepStart = glm::vec3(0.0f);
        double timestamp = 0.0;
        glm::vec3 boundsCenter = glm::vec3(0.0f);
        float boundsRadius = 0.0f;
    };
    struct LayerBrushHistory {
        bool hasBase = false;
        std::vector<uint8_t> baseSnapshot;  // zlib-compressed OctreeFile stream
        uint64_t baseRawBytes = 0;
        std::vector<BrushHistoryEvent> events;
    };
    std::vector<LayerBrushHistory> brushHistories;
    bool undoReplaying = false;
    bool undoKeyWasDown = false;
    // Timestamp slots (12 intervals x 2). Owner CB in brackets; each CB
    // resets only its own slots so cross-queue reset/write races stay
    // impossible:
    //   0/1 .. 4/5 spare (legacy shadow/cull/brush; written by no pass)
    //   6/7 depth prepass [solid]   8/9 sky [solid]   10/11 solid draw [solid]
    //   12/13 vegetation [veg]      14/15 water [water]
    //   16/17 postprocess [main]    18/19 imgui [main]
    //   20/21 RT dispatch [water]   22/23 SDF engine [sdf]
    static constexpr uint32_t QUERY_COUNT = 24;
    std::array<VkQueryPool, MAX_FRAMES_IN_FLIGHT> queryPools = {};
    bool queryPoolReady[MAX_FRAMES_IN_FLIGHT] = {};
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileDepthPrepass = 0.0f;
    float profileSky = 0.0f;
    float profileSolidDraw = 0.0f;
    float profileVegetationImpostor = 0.0f;
    float profileWater = 0.0f;
    float profilePostProcess = 0.0f;
    float profileImGui = 0.0f;
    float profileRTDispatch = 0.0f; // hybrid-RT water pipeline traceRays (slots 20-21)
    float profileSdf = 0.0f;        // SDF engine ray-march pass (slots 22-23)
    // SDF march counters (opt-in, set=1 binding 10): fragment atomics gated by
    // SdfRenderer::setProfileCounters; read back with a 3-frame slot latency.
    bool sdfProfilingEnabled_ = false;
    float profileSdfCpu = 0.0f; // CPU record time of the async SDF task
    // Per-op RT profiling (opt-in): the RT_PROFILE shader variants accumulate
    // per-op counters + device-clock thread-time into per-frame GPU buffers;
    // read/reset one frame-slot behind (see preRenderPass). Off by default —
    // the instrumented pipelines carry atomics and clock reads.
    bool rtProfilingEnabled_ = false;
    RTProfileCounters rtProfileStats_{};
    float profileBackface = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    float profileFps = 0.0f;
#ifdef DEBUG
    uint32_t vramWatchdogCounter_ = 0;
#endif
    UniformObject uboStatic = {};
    // H8 (perf report 22): equirect sky cache. The offscreen sky is
    // view-independent (SkyRendererEquirect.frag reads only SkyUniform + light
    // direction/elevation), so it is re-rendered only when those inputs
    // change; otherwise the previous frame's image stays valid and the task
    // submits nothing, advancing tlSky with a host signal instead. Force the
    // first N frames (the offscreen target is empty until then) and any
    // swapchain-resize recreation.
    static constexpr uint32_t SKY_CACHE_WARMUP = 3; // frames in flight
    uint32_t skyRenderRuns = 0;          // frames actually rendered so far
    float skySig[42] = {};               // last rendered input signature (sky + clouds + quantized time)
    bool skySigValid = false;            // skySig holds a rendered frame's values
    // Guards skySig/skySigValid/skyRenderRuns: the render task commits them
    // from a worker thread after its command buffer begins (see below).
    std::mutex skySigMutex;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<class CloudWidget> cloudWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<GraphicsQualityWidget> graphicsQualityWidget;
    std::shared_ptr<RenderTargetsWidget> renderTargetsWidget;
    std::shared_ptr<ImpostorService> impostorService;
    std::shared_ptr<ImpostorWidget> impostorWidget;
    std::shared_ptr<TextureMixerWidget> textureMixerWidget;
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<CameraWidget> cameraWidget;
    ControllerManager controllerManager;
    std::shared_ptr<ControllerParametersWidget> controllerParametersWidget;
    std::shared_ptr<GamepadWidget> gamepadWidget;
    std::shared_ptr<LightWidget> lightWidget;
    std::shared_ptr<VulkanResourcesManagerWidget> vulkanResourcesManagerWidget;
    std::shared_ptr<QueueTimelineWidget> queueTimelineWidget;
    std::shared_ptr<WindWidget> windWidget;
    // Generic SDF volumes UI (SDF fire is always on; anchors stream in
    // from lava chunks, so no CPU scene mirror lives here). The ray
    // marching widget covers the generic traversal plus the fire and
    // smoke shapes, which are just SDF primitives inside the renderer.
    std::shared_ptr<RaymarchWidget> raymarchWidget;
    std::shared_ptr<GrassWidget> grassWidget;
    std::shared_ptr<MusicWidget> mp3Widget;
    std::shared_ptr<OctreeExplorerWidget> octreeExplorerWidget;
    std::shared_ptr<class LayersWidget> layersWidget;
    std::shared_ptr<RadialMenu> radialMenu;
    std::unique_ptr<RadialMenuHandler> radialMenuHandler;
    WidgetManager widgetManager;
    FilePicker scenePicker_{"Scene File Picker", ".scene"};
    FilePicker blenderPicker_{"Blender Export Picker", ".blend"};
    char blenderExportBuf_[512] = "scenes/octree.blend";
    // Background Blender export (utils/BlenderExport). The worker walks the
    // octree under shared locks; brush edits (unique lock) wait for it.
    std::atomic<bool> blenderExportRunning_{false};
    std::atomic<bool> blenderExportFinished_{false};
    std::atomic<bool> blenderExportCancel_{false};
    std::atomic<float> blenderExportProgress_{0.0f};
    std::mutex blenderExportMutex_;
    std::string blenderExportPhase_;
    blender_export::Stats blenderExportStats_;
    std::thread blenderExportThread_;
    bool blenderExportStatusPending_ = false;
    uint32_t loadedTextureLayers = 0;
    // The startup texture triple list, retained so a resolution-tier change
    // can re-upload the same content (perf report 23 C1).
    std::vector<SceneTextureTriple> textureTriples_;

    // Global texture arrays and material manager (moved from SceneRenderer)
    TextureArrayManager textureArrayManager = TextureArrayManager();
    MaterialManager materialManager = MaterialManager();

    // Texture editing / UI helpers
    std::shared_ptr<TextureMixer> textureMixer;
    // Video textures: every loaded clip owns a slot of the dedicated video
    // texture array; materials of kind Video reference that slot through
    // textureIndex and are rendered by the video layer. Decoding is
    // asynchronous (one worker per clip) and the render loop never waits.
    std::shared_ptr<VideoTextureService> videoTextureService;
    std::shared_ptr<VideoTextureWidget> videoTextureWidget;
    int videoCubeMaterialIndex = -1;   // material used by MainSceneLoader's video cube
    int sdfGivenMaterialIndex = -1;    // material used by MainSceneLoader's SdfGiven cube
    std::shared_ptr<OctreeTopologyCompute> octreeTopologyCompute;
    // ── Water droplets: GPU particle simulation + SDF ray-marched rendering ─
    // The simulation owns device-local state buffers and collides against the
    // cropped octree SDF table (same node layout as the topology pass), so the
    // physics never needs a triangle mesh and never depends on the render mode.
    std::shared_ptr<ParticlePhysicsSim> particleSim;
    std::unique_ptr<DropletRenderer> dropletRenderer;
    std::shared_ptr<class PhysicsWidget> physicsWidget;
    double dropletSeconds = 0.0;    // time since the last re-seed (s)
    double dropletStatsClock = 0.0; // throttles the diagnostics readback
    float dropletShapeRadius = -1.0f; // last uploaded particle radius
    float dropletForcedDt = 0.0f;     // VULKAN_DROPLET_DT override (0 = frame dt)
    // Spawn centre: (0, terrainHeight + dropletSpawnHeight, 0) — a ball of
    // droplets hovering above the height map that then falls onto it.
    glm::vec3 dropletSpawnCenter_{0.0f};
    // Serves THIS application's live scene (Settings::serveScene): the bundled
    // web viewer and --remote engine clients get the chunk ladder plus every
    // live octree edit, same protocol as bin/server.
    std::unique_ptr<SceneServer> sceneServer;
    glm::vec3 dropletRegionMin{0.0f};
    glm::vec3 dropletRegionMax{0.0f};
    uint32_t dropletDrawCount = 0;
    std::vector<MixerParameters> mixerParams;
    std::vector<MaterialProperties> materials;
    // Per-brush image source table (types/ImageType.hpp): brush i -> source
    // (base image / TextureMixer / video). Edited by the Brush widget's mapper.
    BrushMap brushMap;
    // Application-owned per-layer water parameters (initialized in setup)
    std::vector<WaterSettings> waterParams;
    float mainTime = 0.0f;
    // Last frame delta, forwarded to postSubmit for the per-frame brush rebuild
    float lastFrameDelta = 0.0f;
    ShadowParams shadowParams;
    // When user clicks "Apply Brush" from ImGui we defer the octree edit until
    // after the current frame is submitted to avoid waiting on fences while
    // the frame is being recorded (causes deadlock). Set by UI, consumed in
    // `postSubmit()`. (The preview itself streams per frame — no rebuild flag.)
    bool brushApplyToScenePending = false;
    bool generateMapPending = false;
    bool loadScenePending = false;
    bool remoteLoadPending = false;
    std::string pendingLoadPath;

    // Camera and input
    Camera camera = Camera(glm::vec3(-512.0f, 2048.0f, 0.0f), Math::eulerToQuat(-90.0f, 0.0f, 0.0f));
    Light light = Light(glm::vec3(-1.0f, -1.0f, -1.0f));
    EventManager eventManager;
    KeyboardPublisher keyboardPublisher;
    GamepadPublisher gamepadPublisher;
    NunchukPublisher nunchukPublisher;
    MousePublisher mousePublisher;

    // One unified space-change handler per space: each owns both the
    // {onAdded, onDeleted} renderer lambdas and the dedup collector feeding
    // them (scene solid + scene liquid are the two "main" scene spaces).
    // Kept alive as members so the tessellation background thread can safely
    // dispatch them after setup() returns.
    std::thread sceneProcessThread; // tessellates chunks after octree is built

    // Pre-allocated descriptor pool+set rings to avoid per-frame create/destroy
    static constexpr uint32_t ASYNC_RING_SIZE = 3;


    // Dynamic per-scene-layer dispatch: index = Layer. collectors[i] is the
    // dedup stage for layer i; layerAdd/RemoveHandlers[i] are the renderer
    // lambdas built by build(). Rebuilt by ensureLayerHandlers() whenever the
    // active scene's layerCount changes (LayersWidget add/remove). Stored as
    // shared_ptr because UniqueChangeCollector's lambdas capture `this` and
    // the object must stay alive while worker threads dispatch it (add/remove
    // only ever appends/truncates the tail on the main thread).
    std::vector<std::shared_ptr<UniqueChangeCollector>> collectors;
    std::vector<Octree::OctreeNodeDataHandler> layerAddHandlers;
    std::vector<Octree::OctreeNodeDataHandler> layerRemoveHandlers;
    mutable std::mutex layersSyncMutex_;
    // Per-slot resources for the async back-face task, reused in a ring of
    // ASYNC_RING_SIZE slots so the per-frame task allocates nothing.
    // Slot-safety: slot N%ASYNC_RING_SIZE is reused by task N+ASYNC_RING_SIZE.
    // drawFrame waits on the frame fence of frame N before recording frame
    // N+ASYNC_RING_SIZE (ASYNC_RING_SIZE == VulkanApp::MAX_FRAMES_IN_FLIGHT),
    // and frame N's main submission waits on semBackFace (task N), so task N's
    // GPU work -- the only consumer of this slot -- has completed before the
    // slot is touched again. CPU-side tasks also cannot overlap: the main
    // thread blocks on future.get() before enqueueing the next task and the
    // pool has a single worker.
    struct BackfaceSlot {
        VkDescriptorPool pool = VK_NULL_HANDLE;    // per-slot pool (maxSets=1) for back-face set 2
        VkDescriptorSet waterDs = VK_NULL_HANDLE;  // back-face water-depth set (binding 0 = back-face dummy depth)
        VkDescriptorPool poolW = VK_NULL_HANDLE;   // per-slot pool for the parallel water geometry set 2
        VkDescriptorSet waterDs2 = VK_NULL_HANDLE; // water geometry pass set 2 (binding 0 = real backface depth)
    };
    BackfaceSlot cachedBackfaceRing[ASYNC_RING_SIZE]{};
    uint32_t ringBackface = 0;
    ThreadPool asyncThreadPool{1}; // single worker for per-frame back-face pass

    // NOTE (hybrid RT §12): the 360° cubemap capture resources were deleted.
    // Solid/water reflections are hardware ray tracing now.

    ~MyApp() {
        // Cooperative cancel: the exporter checks this between chunks/layers,
        // so closing the app cannot hang on a full-scene export.
        blenderExportCancel_.store(true);
        if (blenderExportThread_.joinable()) blenderExportThread_.join();
    }

    // Rebuild the five material arrays at a new resolution: device idle, drop
    // the old views/images, re-upload the triples, re-point the mixer and
    // refresh its generated layers (perf report 23 C1). Layer indices are
    // resolution-independent, so materials, mixer targets and pickers stay
    // valid; only the texel density changes.
    void recreateTextureArrays(uint32_t size);

    // Prime the ImGui descriptor cache for every map channel of one
    // texture-array layer so the texture viewer can display it immediately.
    void primeImTextureLayer(uint32_t layer) {
        for (int map = 0; map < 5; ++map)
            textureArrayManager.getImTexture(layer, map);
    }

    // setupTextures (defined out-of-line to avoid inline/member-definition issues)
    void setupTextures() {
        // Perf report 23 C1: size the material arrays from content, not a
        // magic 32. The startup path loads this many triples, configures this
        // many mixers, and keeps one editable/paint placeholder plus one
        // spare. The capacity is fixed at bring-up; triples that fail to load
        // only reduce usage below capacity (loadTriples skips them).
        constexpr uint32_t kMinLayers = 8;
        const uint32_t arraySize = static_cast<uint32_t>(std::clamp(settings.textureArraySize, 256, 4096));

        // Shared table (utils/SceneMaterialTable): the same source of truth
        // the Blender exporter reads, so layer indices, texture files and
        // material parameters cannot drift between the renderer and exports.
        textureTriples_ = sceneTextureTriples();
        std::vector<TextureTriple> textureTriples;
        textureTriples.reserve(textureTriples_.size());
        for (const SceneTextureTriple& t : textureTriples_) {
            textureTriples.push_back({t.albedo.c_str(), t.normal.c_str(), t.height.c_str(),
                                      t.rough.c_str(), t.ao.c_str()});
        }

        uint32_t layerCount = static_cast<uint32_t>(textureTriples.size()) + kSceneMixerCount + kScenePaintMargin;
        layerCount = std::max(layerCount, kMinLayers);
        std::cout << "[MyApp::setupTextures] texture arrays: " << layerCount
                  << " layers at " << arraySize << "^2 ("
                  << textureTriples.size() << " triples + " << kSceneMixerCount
                  << " mixers + " << kScenePaintMargin << " paint margin)\n";
        textureArrayManager.allocate(layerCount, arraySize, arraySize, this);

        // Bulk load the triples directly using TextureTriple vector already defined above
        loadedTextureLayers = textureArrayManager.loadTriples(this, textureTriples);
        // Ensure mixer descriptor sets are updated with newly loaded arrays
        
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, &textureArrayManager);
        // Surface Cartesian vertices are extracted only on the GPU (the
        // topology pass writes the packed pools). There is no CPU vertex
        // solver and no readback path to compare against any more.
        octreeTopologyCompute = std::make_shared<OctreeTopologyCompute>();
        octreeTopologyCompute->init(this);
        textureMixer->attachTextureArrayManager(&textureArrayManager);
        // Record into member so UI can display counts
        mixerParams.clear();
        // Generated blend layers (targets continue after the source triples).
        for (const SceneMixerDesc& d : sceneMixerDescs()) {
            mixerParams.push_back(MixerParameters{
                loadedTextureLayers++, d.primary, d.secondary,
                d.perlinScale, d.perlinOctaves, d.perlinPersistence, d.perlinLacunarity,
                d.perlinBrightness, d.perlinContrast, d.perlinSeed, d.perlinTime});
        }

        for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
            primeImTextureLayer(i);
        }

        // Editable placeholder: the first paint-margin layer after the source
        // triples and the generated mixers (video clips live in their own
        // array and do not consume texture-array layers).
        uint32_t editableLayer = (loadedTextureLayers < layerCount) ? loadedTextureLayers : 0u;

        // Trigger initial generation for configured mixers so UI previews show meaningful results
        // (Previously this was deferred to the user pressing "Generate" in the UI)
        std::cerr << "[TextureMixer] Running initial generation for configured mixers...\n";

        // Generate textures for all configured mixer entries (async submissions tracked by TextureMixer)
        textureMixer->generateInitialTextures(mixerParams);
        textureMixerWidget = std::make_shared<TextureMixerWidget>(textureMixer, mixerParams, "Texture Mixer");
        // Shared material editor: define the mix layer's material here.
        textureMixerWidget->setMaterials(&materials);
        textureMixerWidget->setOnMaterialChanged([this](size_t idx) {
            if (idx < materials.size()) materialManager.update(idx, materials[idx], this);
        });
        widgetManager.addWidget(textureMixerWidget);

        // ── Video textures ──────────────────────────────────────────────────
        // Clips live in their own texture array (VideoTextureService) and are
        // loaded at runtime; materials of kind Video reference a slot through
        // textureIndex. MyApp loads the 3 base clips (videos/base/) at setup,
        // the Videos widget can load more from videos/extra/ up to capacity.
        if (std::getenv("VULKAN_NO_VIDEO") == nullptr) {
            videoTextureService = std::make_shared<VideoTextureService>();
            if (videoTextureService->initialize(this, &textureArrayManager)) {
                const std::vector<VideoTextureService::AvailableVideo> available =
                    VideoTextureService::listAvailableVideos();
                size_t loaded = 0;
                for (const auto& clip : available) {
                    if (clip.category != "base") continue;
                    if (videoTextureService->loadVideo(clip.path, clip.name) >= 0) ++loaded;
                    if (loaded >= 3) break;
                }
                if (videoTextureService->empty()) {
                    std::cerr << "[MyApp] no base video could be loaded (see [VideoTexture] log lines)\n";
                    videoTextureService->shutdown();
                    videoTextureService.reset();
                } else {
                    videoTextureWidget = std::make_shared<VideoTextureWidget>(videoTextureService);
                    // Shared material editor: define the selected material's
                    // parameters here, same as the Textures / Texture Mixer panels.
                    videoTextureWidget->setMaterials(&materials);
                    videoTextureWidget->setOnMaterialChanged([this](size_t idx) {
                        if (idx < materials.size()) materialManager.update(idx, materials[idx], this);
                    });
                    widgetManager.addWidget(videoTextureWidget);
                    std::cout << "[MyApp] video textures: " << videoTextureService->count()
                              << " base clip(s) loaded into " << videoTextureService->slotCapacity()
                              << " video slots" << std::endl;
                    if (std::getenv("VULKAN_VIDEO_BENCH") != nullptr) {
                        videoTextureService->runBenchmark(2, 8);
                    }
                }
            } else {
                videoTextureService.reset();
            }
        }

        // Prime the editable placeholder layer (+ its ImGui preview).
        textureArrayManager.setLayerInitialized(editableLayer, true);
        primeImTextureLayer(editableLayer);

        // Reserve material slots for Video-kind materials plus one extra demo
        // material: video materials sample the video array through textureIndex
        // instead of a material texture layer, so they need no layer in the five
        // material arrays. MainSceneLoader's cubes paint with these slots.
        const size_t kVideoMaterialCount = 4;
        const size_t kExtraMaterialCount = kVideoMaterialCount + 1;
        size_t materialCount = static_cast<size_t>(loadedTextureLayers) + 1 + kExtraMaterialCount;
        // Same per-layer overrides the Blender exporter publishes.
        materials = sceneMaterialProperties(materialCount);
        for (size_t i = static_cast<size_t>(loadedTextureLayers) + 1;
             i < static_cast<size_t>(loadedTextureLayers) + 1 + kVideoMaterialCount; ++i) {
            MaterialProperties& m = materials[i];
            m.kind = MaterialKind::Video;
            m.textureIndex = static_cast<int>(i - (static_cast<size_t>(loadedTextureLayers) + 1));
            m.blendMode = MaterialBlendMode::Normal;
        }
        // First video material: the demo cube, spherical SDF coordinates around
        // its bounding volume and Screen blending (Solid -> Water -> Video).
        if (videoTextureService) {
            MaterialProperties& videoMat = materials[static_cast<size_t>(loadedTextureLayers) + 1];
            videoMat.kind = MaterialKind::Video;
            videoMat.textureIndex = 0;
            videoMat.mappingOrigin = glm::vec4(0.0f, 512.0f, 512.0f * 8.0f, 320.0f);
            videoMat.blendMode = MaterialBlendMode::Screen;
            videoCubeMaterialIndex = static_cast<int>(loadedTextureLayers) + 1;
        }
        // Extra demo material: a Texture-kind material using Angular spherical
        // coordinates around the solid demo cube's bounding volume.
        {
            const size_t sdfIndex = static_cast<size_t>(loadedTextureLayers) + 1 + kVideoMaterialCount;
            MaterialProperties& m = materials[sdfIndex];
            m.kind = MaterialKind::Texture;
            m.textureIndex = 6;  // existing decorative texture layer
            m.mappingOrigin = glm::vec4(0.0f, 512.0f, 512.0f * 9.0f, 320.0f);
            m.blendMode = MaterialBlendMode::Normal;
            sdfGivenMaterialIndex = static_cast<int>(sdfIndex);
        }

        // Headless blend-mode A/B: force the video cube material's blend mode
        // from the command line (VULKAN_VIDEO_BLEND=<index 0..18>). The normal
        // upload below picks it up with the rest of the materials.
        if (videoCubeMaterialIndex >= 0) {
            if (const char* blendEnv = std::getenv("VULKAN_VIDEO_BLEND")) {
                const int mode = std::atoi(blendEnv);
                if (mode >= 0 && mode <= static_cast<int>(MaterialBlendMode::Luminosity)) {
                    materials[static_cast<size_t>(videoCubeMaterialIndex)].blendMode =
                        static_cast<MaterialBlendMode>(mode);
                }
            }
        }

        // Allocate GPU-side material storage via MaterialManager
        materialManager.allocate(materialCount, this);
        for (size_t i = 0; i < materialCount; ++i) materialManager.update(i, materials[i], this);

        // Per-brush image map. Default: brush i samples base image layer i. The
        // TextureMixer outputs and the video slots get their own entries so the
        // renderer (and the widgets) can tell the three sources apart. Sized
        // generously (brush indices may exceed the material count, e.g. via the
        // controller's material field) so the shader lookup can never run off
        // the end of the table.
        const size_t brushMapCount = std::max<size_t>(materialCount, 64);
        brushMap.resize(brushMapCount);
        for (size_t i = 0; i < brushMapCount; ++i)
            brushMap[i] = BrushImage{ImageType::Image, static_cast<uint32_t>(i)};
        const size_t imageCount = textureTriples_.size();  // base image layers
        for (size_t k = 0; k < mixerParams.size(); ++k) {
            const uint32_t layer = static_cast<uint32_t>(imageCount + k);
            brushMap[layer] = BrushImage{ImageType::TextureMixer, layer};
        }
        for (size_t k = 0; k < kVideoMaterialCount; ++k) {
            const size_t brush = static_cast<size_t>(loadedTextureLayers) + 1 + k;
            brushMap[brush] = BrushImage{ImageType::Video, static_cast<uint32_t>(k)};
        }
        if (sdfGivenMaterialIndex >= 0)
            brushMap[static_cast<size_t>(sdfGivenMaterialIndex)] = BrushImage{ImageType::Image, 6};
        materialManager.allocateBrushMap(brushMapCount, this);
        for (size_t i = 0; i < brushMapCount; ++i) materialManager.updateBrushMap(i, brushMap[i], this);

        // The descriptor set was bound to a dummy buffer at SceneRenderer::init() time
        // because materialManager hadn't been allocated yet (runs on a background thread).
        // Now that the real buffer exists, rebind descriptor set binding 5 so the GPU
        // reads from the actual materials SSBO.
        if (sceneRenderer) {
            sceneRenderer->updateTextureDescriptorSet(this, &textureArrayManager);
        }

        // Perf report 23 C1: committed vs. initialized texture memory. The
        // scene renderer logs it again after every scene load/generation.
        textureArrayManager.logMemoryUtilization("main");

    }

     void setup() override {
        applySetupEnvOverrides();

        // VULKAN_SETUP_DIAG=1: print the wall time of each top-level setup step
        // so the parallelisation work can be measured.
        const bool setupDiag = std::getenv("VULKAN_SETUP_DIAG") != nullptr;
        auto setupDiagLast = std::chrono::steady_clock::now();
        auto setupStep = [&](const char* name) {
            if (!setupDiag) return;
            const auto now = std::chrono::steady_clock::now();
            std::cout << "[setup-diag] " << name << ": "
                      << std::chrono::duration<double, std::milli>(now - setupDiagLast).count()
                      << " ms" << std::endl;
            setupDiagLast = now;
        };

        createSceneInfrastructure(setupStep);
        createWorldAndBrushes();
        initSceneRenderers(setupStep);
        createTextureViewerPanel();
        createUiWidgets();
        registerUiWidgets();
        setupInputAndCamera();
        createBrushAndLayerWidgets();
        createProfilingResources();
        scheduleInitialSceneLoad();
        playSetupAlert("setup-ready");
    }

    // ── setup() sections ─────────────────────────────────────────────────────
    void applySetupEnvOverrides() {
        // Headless droplet bring-up (validation/diagnostics): VULKAN_DROPLETS=1
        // enables the simulation + rendering and the counter readback — the
        // same state the Droplets widget produces.
        if (const char* env = std::getenv("VULKAN_DROPLETS")) {
            if (env[0] != '0') {
                settings.dropletsEnabled = true;
                settings.dropletStats = true;
            }
        }
        if (const char* env = std::getenv("VULKAN_DROPLET_COUNT")) {
            settings.dropletCount = std::max(1, std::min(std::atoi(env), 8192));
        }
        // Headless brush-preview bring-up (composite validation): add one
        // default brush entry so the preview overlay renders without UI input.
        // VULKAN_BRUSH_SCALE optionally overrides its uniform scale (a large
        // value spans the visible water surface for depth-order checks).
        if (std::getenv("VULKAN_BRUSH_PREVIEW") != nullptr) {
            brushManager.addEntry();
        }

        // VULKAN_SERVE=1: start the scene server for this app's live scene
        // (same as the Settings widget toggle).
        if (const char* env = std::getenv("VULKAN_SERVE")) {
            if (env[0] != '0') settings.serveScene = true;
        }

        // VULKAN_CAMERA="x,y,z,yaw,pitch,roll": override the start pose
        // (headless diagnostics, e.g. a top-down view that makes plane-shaped
        // mesh gaps visible). Angles in degrees, same convention as
        // Math::eulerToQuat.
        if (const char* cam = std::getenv("VULKAN_CAMERA")) {
            float x = 0.0f, y = 0.0f, z = 0.0f, yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
            if (std::sscanf(cam, "%f,%f,%f,%f,%f,%f", &x, &y, &z, &yaw, &pitch, &roll) == 6) {
                camera = Camera(glm::vec3(x, y, z), Math::eulerToQuat(yaw, pitch, roll));
            }
        }

        // Headless A/B switches for layer isolation (diagnostics only): each
        // forces the matching Settings toggle off/on before the first frame.
        if (std::getenv("VULKAN_NO_WATER") != nullptr) settings.waterEnabled = false;
        if (std::getenv("VULKAN_NO_SOLID") != nullptr) settings.renderSolid = false;
        if (std::getenv("VULKAN_NO_VEG") != nullptr) settings.vegetationEnabled = false;
        if (std::getenv("VULKAN_WIREFRAME") != nullptr) settings.wireframeMode = true;

        // VULKAN_DROPLET_DT forces a fixed per-frame simulation delta (headless
        // validation on slow software rasterizers, where the real frame delta
        // is clamped to 0.1 s and the fall would take minutes of wall time).
        if (const char* env = std::getenv("VULKAN_DROPLET_DT")) {
            dropletForcedDt = std::max(0.0f, static_cast<float>(std::atof(env)));
        }
        // VULKAN_DROPLET_RADIUS overrides the particle radius (visibility in
        // low-resolution captures).
        if (const char* env = std::getenv("VULKAN_DROPLET_RADIUS")) {
            settings.dropletRadius = std::max(0.05f, static_cast<float>(std::atof(env)));
        }
    }

    void createSceneInfrastructure(const std::function<void(const char*)>& step) {
        // Build the scene descriptor layouts/sets. Owned by the application (not by
        // the generic VulkanApp framework) so the engine core stays agnostic about
        // scene bindings. Must exist before SceneRenderer::init() writes the static
        // descriptor set below.
        sceneDescriptorLayout = std::make_unique<SceneDescriptorLayout>();
        sceneDescriptorLayout->create(*this);
        playSetupAlert("setup-layout");
        step("scene descriptor layout");

        // Build the scene render queues from the parallel graphics-family queue
        // pool built by VulkanApp::createLogicalDevice. They alias graphicsQueue
        // when the device exposes fewer physical graphics queues. Owned via the
        // generic VulkanApp::sceneQueues member (forward-declared; the framework
        // only forwards), so the engine core stays agnostic about scene staging.
        sceneQueues = std::make_unique<SceneQueues>();
        sceneQueues->configure(getParallelGraphicsQueues(), getGraphicsQueue());

        sceneRenderer = new SceneRenderer();
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i)
            shadowParams.shadowMapSizes[i] = sceneRenderer->shadowMapper->getShadowMapSize(i);
        // Initialize application-owned water params with two default elements
        {
            // First (default) water material: shore waves enabled. All other
            // layers keep the struct default (enableWaves = false), i.e. calm.
            WaterSettings wp0 = WaterSettings();
            wp0.enableWaves = true;
            waterParams.push_back(wp0);
        }
        {
            WaterSettings wp = WaterSettings();
            wp.noiseOctaves = 1;
            wp.causticColor = glm::vec3(1.0f, 0.98f, 0.9f); // sunlight tint
            // One water region: a single green tint for this demo layer.
            wp.waterColor = glm::vec3(0.03f, 0.30f, 0.12f);
            wp.waterTint = 0.6f;
            wp.causticIntensity = 0.2f;
            wp.causticSoftness = 0.5f;
            wp.tessMinLevel = 2.0f;
            wp.tessMaxLevel = 16.0f;
            // Polished mirror like layer 2: a 100% mirror transmits nothing,
            // so refraction is off (it would only be masked out anyway) and
            // the partial-mirror shoreline dissolve never applies.
            wp.enableRefraction = false;
            wp.reflectionStrength = 1.0f;
            wp.fresnelPower = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }
        {
            WaterSettings wp = WaterSettings();
            wp.enableRefraction = false;
            wp.noiseOctaves = 0;
            wp.noisePeriod = 0.0f;
            wp.causticColor = glm::vec3(1.0f, 1.0f, 1.0f);
            // White -> black depth-region ramp for this stylized demo layer.
            wp.waterColor = glm::vec3(0.04f, 0.22f, 0.36f); // one water region
            wp.waterTint = 1.0f;
            wp.causticIntensity = 0.0f; // caustics off on this layer
            wp.waveAmplitude = 0.0f;
            wp.blurRadius = 4.0f;
            wp.enableBlur = true;
            wp.reflectionStrength = 1.0f;
            wp.fresnelPower = 1.0f;
            wp.tessMinLevel = 1.0f;
            wp.tessMaxLevel = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }
    }

    void createWorldAndBrushes() {
        // Create scene objects and build the octree synchronously in setup().
        // Chunk tessellation is deferred and processed on a background thread.
        // Vulkan GPU uploads happen on the main thread via processPendingMeshes().
        scenePicker_.addBookmark(
            reinterpret_cast<const char*>(u8"\uf07c##scene_bm_scenes"),
            "Go to project scenes folder",
            std::filesystem::path("scenes"));

        // Create the World (owns Chunks, Octrees, and the ChunkManager state machine).
        // The renderer receives a reference to the World for chunk state queries
        // and proxy swap notifications.
        world = new World();
        sceneRenderer->setWorld(world);

        // Try the remote chunk server first; on any failure the local octrees
        // stay authoritative (see useRemote guards below). The octree explorer
        // always shows the LOCAL scene (empty in remote mode) — remote chunks
        // live in RemoteScene stub nodes, not in an octree.
        if (!remoteDisabled) {
            RemoteScene::Endpoint ep;
            if (parseRemoteEndpoint(remoteEndpoint, ep)) {
                auto remote = std::make_unique<RemoteScene>();
                if (remote->connect(ep)) {
                    useRemote = true;
                    remoteScene = std::move(remote);
                    world->setActiveScene(remoteScene.get());
                    std::cout << "[MyApp::setup] rendering RemoteScene from "
                              << ep.host << ":" << ep.port << "\n";
                }
            } else {
                std::cout << "[MyApp::setup] bad remote endpoint '" << remoteEndpoint
                          << "'; keeping LocalScene\n";
            }
        }
        if (!useRemote) {
            std::cout << "[MyApp::setup] rendering LocalScene\n";
        }
        playSetupAlert("setup-world");

        octreeExplorerWidget = std::make_shared<OctreeExplorerWidget>(&world->scene(), &camera);
        widgetManager.addWidget(octreeExplorerWidget);
        brushManager.getEntries().clear();
        brushManager.getEntries().resize(3);
        brushManager.getEntries()[0].sdfType = 1;
        brushManager.getEntries()[0].materialIndex = 0;
        brushManager.getEntries()[0].translate = glm::vec3(0.0f, 0.0f, 0.0f);
        brushManager.getEntries()[0].scale = glm::vec3(256.0f);
        brushManager.getEntries()[0].hsv = glm::vec3(0.0f, 0.5f, 0.5f);
        brushManager.getEntries()[1].sdfType = 1;
        brushManager.getEntries()[1].materialIndex = 1;
        brushManager.getEntries()[1].translate = glm::vec3(512.0f, 0.0f, 0.0f);
        brushManager.getEntries()[1].scale = glm::vec3(256.0f);
        brushManager.getEntries()[1].hsv = glm::vec3(120.0f, 0.8f, 1.0f);
        brushManager.getEntries()[2].sdfType = 3;
        brushManager.getEntries()[2].materialIndex = 2;
        brushManager.getEntries()[2].translate = glm::vec3(-512.0f, 0.0f, 0.0f);
        brushManager.getEntries()[2].scale = glm::vec3(256.0f);
        brushManager.getEntries()[2].hsv = glm::vec3(240.0f, 0.7f, 1.0f);
        // minSize = tessellation frontier (MainSceneLoader default 30); the
        // octree walk emits cells at every ladder level (chunkLod 1..5) and
        // the GPU cull keeps the level matching the camera distance.
        // build() creates the {onAdded, onDeleted} renderer lambdas for each
        // scene layer; the dedup collectors in front of them (fed to
        // Scene::loadScene/action and Octree::apply) replay final per-node
        // state into these handlers on the tessellation threads.
        // sceneForChanges is the ACTIVE scene (remote stubs or local octree).
        ensureLayerHandlers();

        // Scene starts empty — use File > Generate Map to populate it.
    }

    void initSceneRenderers(const std::function<void(const char*)>& step) {
        // Init the VegetationRenderer before setupVegetationTextures so that
        // wind params UBO + descriptor set layout exist before captureAll calls
        // setImpostorData().  SceneRenderer::init() is called later after all
        // texture/material setup is complete.
        if (sceneRenderer && sceneRenderer->vegetationRenderer)
            sceneRenderer->vegetationRenderer->init(this);
        step("vegetation renderer init");

        setupVegetationTextures();
        step("vegetation textures");
        setupTextures();
        playSetupAlert("setup-textures");
        step("scene textures + materials");

        // H9: the water offscreen targets render at Settings::waterRenderScale.
        sceneRenderer->setWaterRenderScale(settings.waterRenderScale);
        // M12 (perf report 22): the vegetation offscreen targets render at
        // Settings::vegetationRenderScale.
        sceneRenderer->setVegetationRenderScale(settings.vegetationRenderScale);
        sceneRenderer->init(this, &textureArrayManager, &materialManager, waterParams);
        step("scene renderer init (renderers, pipelines, descriptors)");

        // H3 (perf report 29): seed the debug collection gates from the settings
        // before any chunk is built, so the default (both overlays off) never
        // runs the SDF-cube / bounding-box subtree walks or retains their
        // records. Runtime toggles are applied per frame in postSubmit().
        applyDebugOverlaySettings();

        // Bind the video texture array now that both the service (setupTextures)
        // and the video layer renderer exist.
        if (sceneRenderer->videoRenderer) {
            sceneRenderer->videoRenderer->setVideoTexture(this, videoTextureService.get());
        }

        // Re-wire impostors now that VegetationRenderer::init() has stored the render pass.
        if (impostorService) impostorService->rewire();

        printf("[MyApp::setup] Created and initialized SceneRenderer\n");
        playSetupAlert("setup-renderers");
        step("renderer post-init (video bind, impostor rewire)");
    }

    void createTextureViewerPanel() {
        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(&textureArrayManager, &materials);
        textureViewer->setImageCount(static_cast<uint32_t>(textureTriples_.size()));
        textureViewer->setVideoSlotNamesProvider([this]() {
            std::vector<std::string> names;
            if (!videoTextureService) return names;
            const uint32_t capacity = videoTextureService->slotCapacity();
            names.resize(capacity);
            for (const auto& s : videoTextureService->states()) {
                if (s.slot >= capacity) continue;
                char label[128];
                std::snprintf(label, sizeof(label), "%u: %s", s.slot, s.name.c_str());
                names[s.slot] = label;
            }
            for (uint32_t i = 0; i < capacity; ++i) {
                if (names[i].empty()) names[i] = std::to_string(i) + ": <empty>";
            }
            return names;
        });
        textureViewer->setVideoService(videoTextureService.get());
        textureViewer->setVideoMaterialIndex(videoCubeMaterialIndex);
        textureViewer->setOnFocusVideoCube([this]() {
            // In front of the demo cube at (0,512,4096); yaw 180 faces +Z.
            camera = Camera(glm::vec3(0.0f, 700.0f, 3300.0f), Math::eulerToQuat(180.0f, 0.0f, 0.0f));
        });
        textureViewer->setVideoPreviewProvider([this](uint32_t slot, int map) -> ImTextureID {
            if (!videoTextureService) return 0;
            switch (map) {
                case 0: return videoTextureService->previewTexture(slot);
                case 1: return videoTextureService->derivedPreviewTexture(slot, 0); // normal
                case 2: return videoTextureService->derivedPreviewTexture(slot, 1); // height/bump
                case 3: return videoTextureService->derivedPreviewTexture(slot, 2); // roughness/spec
                default: return 0;
            }
        });
        textureViewer->setOnMaterialChanged([this](size_t idx) {
            materialManager.update(idx, materials[idx], this);
        });
    }

    void createUiWidgets() {
        skyWidget = std::make_shared<SkyWidget>(sceneRenderer->getSkySettings());
        cloudWidget = std::make_shared<CloudWidget>(sceneRenderer->getCloudSettings());
        physicsWidget = std::make_shared<PhysicsWidget>(settings);
        widgetManager.addWidget(physicsWidget);
        // Create settings widget (was missing previously)
        settingsWidget = std::make_shared<SettingsWidget>(settings, &shadowParams);
        // Water UI uses the application-owned water params vector and updates GPU state explicitly.
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer->waterRenderer.get(), &waterParams);

        // Right-aligned main-UI preset buttons (publishes SetGraphicsQualityEvent).
        graphicsQualityWidget = std::make_shared<GraphicsQualityWidget>(&eventManager);

        renderTargetsWidget = std::make_shared<RenderTargetsWidget>(
            this,
            sceneRenderer, sceneRenderer->solidRenderer.get(), sceneRenderer->skyRenderer.get(),
            sceneRenderer->shadowMapper.get(), &shadowParams);
        if (renderTargetsWidget) renderTargetsWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());

        cameraWidget = std::make_shared<CameraWidget>(&camera);
        controllerParametersWidget = std::make_shared<ControllerParametersWidget>(&controllerManager, &brushManager);
        gamepadWidget = std::make_shared<GamepadWidget>(&controllerManager, &nunchukPublisher);
        // Auto-connect to a Wiimote (with or without Nunchuk) on startup.
        nunchukPublisher.connect();
        lightWidget = std::make_shared<LightWidget>(&light);
        vulkanResourcesManagerWidget = std::make_shared<VulkanResourcesManagerWidget>(&resources);
        vulkanResourcesManagerWidget->updateWithApp(this);
        queueTimelineWidget = std::make_shared<QueueTimelineWidget>(this);
        queueTimelineWidget->updateWithApp(this);
        windWidget = std::make_shared<WindWidget>(sceneRenderer->vegetationRenderer.get());
        raymarchWidget = std::make_shared<RaymarchWidget>(
            sceneRenderer->sdfRenderer.get(), &camera,
            sceneRenderer->vegetationRenderer.get());
        // Bladed grass tufts (triangle blades + captured impostors).
        grassWidget = std::make_shared<GrassWidget>(sceneRenderer->vegetationRenderer.get());
        // Generic SDF fire volume (spec §19) is the fire path; flame
        // anchors stream in from brush-4 lava chunks as they publish (see
        // SceneRenderer::processPendingMeshes ingest hook).
        if (sceneRenderer->sdfRenderer) {
            sceneRenderer->sdfRenderer->setRenderMode(SdfRenderer::RenderMode::Volume);
        }
        mp3Widget = std::make_shared<MusicWidget>();

        // Radial menu (input-agnostic overlay, not a Widget subclass)
        radialMenu = std::make_shared<RadialMenu>();
        radialMenuHandler = std::make_unique<RadialMenuHandler>(
            getWindow(), &eventManager, radialMenu.get(),
            &nunchukPublisher, &gamepadPublisher, &controllerManager,
            &brushManager, &textureArrayManager, &light);
        radialMenuHandler->setupPages();
  // Create octree explorer widget bound to loaded scene
    }

    void registerUiWidgets() {
        widgetManager.addWidget(textureViewer);
        widgetManager.addWidget(cameraWidget);
        widgetManager.addWidget(controllerParametersWidget);
        widgetManager.addWidget(gamepadWidget);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(graphicsQualityWidget);
        widgetManager.addWidget(lightWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(cloudWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderTargetsWidget);
        widgetManager.addWidget(vulkanResourcesManagerWidget);
        widgetManager.addWidget(queueTimelineWidget);
        widgetManager.addWidget(windWidget);
        widgetManager.addWidget(raymarchWidget);
        widgetManager.addWidget(grassWidget);
        widgetManager.addWidget(mp3Widget);
        widgetManager.addWidget(impostorWidget);
    }

    void setupInputAndCamera() {
        // Subscribe event handlers
        eventManager.subscribe(&camera);  // Camera handles translate/rotate events
        eventManager.subscribe(this);     // MyApp handles close/fullscreen events

        // Mouse input needs the GLFW window to poll cursor / chain scroll.
        mousePublisher.attachWindow(getWindow());
        
        // Set up camera projection matrix
        float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, settings.nearPlane, settings.farPlane);
        proj[1][1] *= -1; // Vulkan Y-flip
        camera.setProjection(proj);
        shadowParams.update(camera.getPosition(), light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);
        
        // Position camera to view the terrain
        printf("[Camera Setup] Final Position: (%.1f, %.1f, %.1f)\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
        printf("[Camera Setup] Forward: (%.3f, %.3f, %.3f)\n", camera.getForward().x, camera.getForward().y, camera.getForward().z);
    }

    void createBrushAndLayerWidgets() {
        // Create brush3dWidget after setupTextures() so loadedTextureLayers is set.
        brush3dWidget = std::make_shared<Brush3dWidget>(&textureArrayManager, loadedTextureLayers, brushManager, &eventManager);
        brush3dWidget->setScene(&world->activeScene());
        // Brush mapper: per-brush image source (Image / TextureMixer / Video).
        brush3dWidget->setBrushMap(&brushMap, static_cast<uint32_t>(textureTriples_.size()),
                                   static_cast<uint32_t>(mixerParams.size()));
        brush3dWidget->setOnBrushMapChanged([this](size_t idx) {
            if (idx < brushMap.size()) materialManager.updateBrushMap(idx, brushMap[idx], this);
        });
        widgetManager.addWidget(brush3dWidget);

        // Dynamic scene layers + per-layer renderer selection.
        layersWidget = std::make_shared<LayersWidget>(&world->activeScene(), sceneRenderer,
            [this]() { syncLayers(); },
            [this](Layer layer) { repopulateLayer(layer); });
        widgetManager.addWidget(layersWidget);
        playSetupAlert("setup-ui");
    }

    void createProfilingResources() {
        // Create per-frame timestamp query pools for GPU profiling
        {
            VkPhysicalDeviceProperties physProps{};
            vkGetPhysicalDeviceProperties(getPhysicalDevice(), &physProps);
            timestampPeriod = physProps.limits.timestampPeriod;
            if (timestampPeriod > 0.0f) {
                VkQueryPoolCreateInfo qpci{};
                qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qpci.queryCount = QUERY_COUNT;
                for (uint32_t f = 0; f < 3; ++f) {
                    if (vkCreateQueryPool(getDevice(), &qpci, nullptr, &queryPools[f]) != VK_SUCCESS)
                        throw std::runtime_error("Failed to create timestamp query pool");
                }
                // Pre-reset every slot once (setup path may block): a discarded
                // async task (exception before its in-CB reset records) would
                // otherwise leave pristine queries behind, tripping
                // "query not reset" on the first readback. Steady-state resets
                // stay in their owning command buffers (no cross-queue races).
                runSingleTimeCommands([&](VkCommandBuffer cmd) {
                    for (uint32_t f = 0; f < 3; ++f)
                        if (queryPools[f] != VK_NULL_HANDLE)
                            vkCmdResetQueryPool(cmd, queryPools[f], 0, QUERY_COUNT);
                });
            }
            // Stall hook: when drawFrame detects a GPU ring hang (frame-slot
            // fence wait >0.5s or stale frame timeline), read every pool with
            // PARTIAL_BIT (no wait) — the pool of the stuck frame has start
            // timestamps written but its end timestamp missing, which names the
            // exact pass the GPU is stuck in. Also dump the submission ring.
            onFrameStall = [this](uint32_t) {
                // One name per timestamp interval (2 slots each): interval 3 is
                // slots 6/7 (depth) ... interval 11 is slots 22/23 (SDF).
                static const char* intervalNames[9] = {
                    "depth", "sky", "solid", "veg", "water", "post", "imgui", "rt", "sdf"
                };
                for (uint32_t f = 0; f < 3; ++f) {
                    if (queryPools[f] == VK_NULL_HANDLE) continue;
                    // Timestamp query pools may NOT be read with VK_QUERY_RESULT_PARTIAL_BIT
                    // (VUID-vkGetQueryPoolResults-queryType-09439). Use WITH_AVAILABILITY_BIT
                    // instead: each query result is a (value, availability) pair, so the stride
                    // is two uint64_t per query and we test availability to detect a stuck pass.
                    std::array<uint64_t, QUERY_COUNT * 2> ts{};
                    if (vkGetQueryPoolResults(getDevice(), queryPools[f], 0, QUERY_COUNT,
                            sizeof(ts), ts.data(), sizeof(uint64_t) * 2,
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS)
                        continue;
                    std::cerr << "[stall] pool " << f << " timestamps:\n";
                    for (uint32_t i = 3; i < 12; ++i) {
                        const uint64_t startVal = ts[4 * i], startAvail = ts[4 * i + 1];
                        const uint64_t endVal = ts[4 * i + 2], endAvail = ts[4 * i + 3];
                        const bool haveStart = startAvail != 0 && startVal != 0;
                        const bool haveEnd = endAvail != 0 && endVal != 0;
                        std::cerr << "[stall]   " << intervalNames[i - 3]
                                  << (haveEnd ? " DONE " : (haveStart ? " STUCK " : "  idle "))
                                  << " start=" << startVal << " end=" << endVal << "\n";
                    }
                }
            };
        }
    }

    void scheduleInitialSceneLoad() {
        // Scene content source: remote snapshot when connected, otherwise the
        // local default scene file or procedural generation.
        const std::string defaultScenePath = "scenes/default.scene";
        if (useRemote) {
            remoteLoadPending = true;
            std::cout << "[MyApp::setup] Loading chunks from remote server on first frame\n";
        } else if (std::filesystem::exists(defaultScenePath)) {
            pendingLoadPath = defaultScenePath;
            loadScenePending = true;
            std::cout << "[MyApp::setup] Loading default scene from '" << defaultScenePath << "'\n";
        } else {
            generateMapPending = true; // Trigger initial map generation on first frame
            std::cout << "[MyApp::setup] No default scene found, generating procedural map\n";
        }
    }


    // Move vegetation texture setup into its own method for clarity
    void setupVegetationTextures();
    // Stream the selected brush entry's SDF description into the per-frame
    // BrushSdfUBO (preview raymarch + solid PAINT/REMOVE intersection).
    void updateBrushPreview(uint32_t frameIndex);
    // Apply the selected brush SDF to the main scene's octree on the selected layer
    void applyBrushToScene();
    // Apply one brush event to a layer's octree (shared by live apply and undo
    // replay); does not record history or advance previousTranslate.
    void applyBrushEventToLayer(const BrushEntry& entry, const glm::vec3& sweepStart, Layer layer);
    // Serialize/restore the layer base and manage the per-layer brush stack.
    void ensureBrushBase(Layer layer, Octree& tree);
    void recordBrushEvent(Layer layer, const BrushHistoryEvent& event);
    void undoBrushLayer(Layer layer, bool group);
    void clearBrushHistory();
    void pollBrushUndo();
    // Clear GPU meshes, reset octrees and regenerate via MainSceneLoader
    void generateMap();
    // Builds the GPU droplet sim + renderer for the active scene (once).
    void ensureDroplets();
    void action();
    // Clear GPU meshes, reset octrees, load from file and tessellate
    void loadSceneFromFile(const std::string& path);
    // Shared scene state reset (join thread, wait GPU, clear meshes/octrees/handlers)
    void resetSceneState();
    // Replay the deduplicated scene change events into the renderer handlers
    // (replaces UniqueOctreeChangeHandler::handleEvents).
    void dispatchLayerEvents(Layer layer);
    void dispatchAllEvents();
    // Rebuild collectors + renderer handlers for the active scene's current
    // layer list. Safe to call any time on the main thread: grows
    // SceneRenderer states + collectors, rebuilds handlers only when the
    // count changed (existing handlers stay valid — SceneRenderer states are
    // heap-allocated).
    void ensureLayerHandlers();
    // Sync after LayersWidget add/remove: ensure handlers + clear GPU for
    // removed layers. Called from update() (poll) and explicitly by the widget.
    void syncLayers();
    // Re-queue one layer's chunks for tessellation (used after re-enable or
    // renderer switch). Re-emits the layer into its collector and dispatches
    // it on a worker thread; GPU upload follows progressively on the main
    // thread. Safe to call from the UI thread.
    void repopulateLayer(Layer layer);

    // ── H3 (perf report 29): debug overlay collection gate ─────────────────
    // Applies Settings::showSDFDebug / showBoundingBoxes to the debug
    // renderers' collection bits (startup + every frame, so a widget toggle is
    // picked up). The draw path keeps checking the settings itself; this only
    // starts/stops the subtree walks and their retained records. A false->true
    // transition runs one re-collection pass so the overlay is complete
    // immediately. Main thread only.
    void applyDebugOverlaySettings();
    // One-shot re-collection of both debug overlays for every live chunk
    // (no re-tessellation, no upload). See applyDebugOverlaySettings().
    void recollectDebugOverlays();

    // Start the background tessellation thread for the current scene. Every
    // layer dispatches on its own thread so all layers tessellate truly in
    // parallel. `completionLog` is printed verbatim once all threads join.
    void startTessellationThreads(const char* completionLog) {
        ensureLayerHandlers();
        size_t n = 0;
        {
            std::lock_guard<std::mutex> lock(layersSyncMutex_);
            n = collectors.size();
        }
        sceneProcessThread = std::thread([this, completionLog, n]() {
            std::vector<std::thread> workers;
            workers.reserve(n);
            for (size_t i = 0; i < n; ++i)
                workers.emplace_back([this, i]() { dispatchLayerEvents(static_cast<Layer>(i)); });
            for (auto& t : workers) t.join();
            std::cout << completionLog;
        });
    }

    // Parse "host:port" into a RemoteScene endpoint. Returns false on garbage.
    static bool parseRemoteEndpoint(const std::string& s, RemoteScene::Endpoint& out) {
        const size_t colon = s.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= s.size()) return false;
        try {
            const int port = std::stoi(s.substr(colon + 1));
            if (port <= 0 || port > 65535) return false;
            out.host = s.substr(0, colon);
            out.port = static_cast<uint16_t>(port);
            return !out.host.empty();
        } catch (...) {
            return false;
        }
    }

    // Download the remote snapshot into the collectors. On failure the caller
    // falls back to LocalScene (collectors are cleared first).
    void loadRemoteScene();
    // Recurring dispatch loop for the live remote stream. Reuses
    // sceneProcessThread (joined by the existing reset/shutdown paths).
    // The loop runs while remoteDispatchRun is set; stopRemoteDispatch()
    // flips it and joins (safe to call in local mode too: no-op).
    std::atomic<bool> remoteDispatchRun{false};
    void startRemoteDispatchLoop() {
        ensureLayerHandlers();
        remoteDispatchRun = true;
        sceneProcessThread = std::thread([this]() {
            std::cout << "[MyApp] remote dispatch loop started\n";
            while (remoteDispatchRun.load()) {
                dispatchAllEvents();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            std::cout << "[MyApp] remote dispatch loop stopped\n";
        });
    }
    void stopRemoteDispatch() {
        if (remoteDispatchRun.exchange(false)) {
            if (sceneProcessThread.joinable()) sceneProcessThread.join();
        }
    }
    // Disconnect the remote scene first (wakes mesh fetchers blocked on the
    // socket), before any thread joins or pool stops below.
    void stopRemoteStreaming() {
        stopRemoteDispatch();
        if (remoteScene) remoteScene->disconnect();
    }

// (setup implementation defined out-of-line below)

    void update(float deltaTime) override {

        if (deltaTime > 0.0f) profileFps = 1.0f / deltaTime;
        auto cpuUpdateT0 = std::chrono::high_resolution_clock::now();

        // Suppress all normal input when radial menu is visible
        bool radialMenuVisible = radialMenu && radialMenu->IsVisible();

        if (!radialMenuVisible) {
            keyboardPublisher.update(getWindow(), &eventManager, camera, deltaTime, &controllerManager, &brushManager, false);
            gamepadPublisher.update(&eventManager, camera, deltaTime, &controllerManager, &brushManager, false);
            nunchukPublisher.update();
            nunchukPublisher.applyControls(&eventManager, camera, deltaTime,
                                           &controllerManager, &brushManager,
                                            (world && !useRemote && world->scene().layerCount() > 0) ? world->scene().getLayerOctree(0) : nullptr);
        } else {
            // Still poll nunchuk state so Home/A button edge detection works
            nunchukPublisher.update();
            // Poll gamepad left stick for radial menu input
            gamepadPublisher.pollLeftStick();
        }

        // Mouse: suppress when radial menu is visible or ImGui captures mouse
        bool mouseSuppressed = ImGui::GetIO().WantCaptureMouse || radialMenuVisible;
        mousePublisher.update(&eventManager, camera, &controllerManager,
                             &brushManager, mouseSuppressed);
        eventManager.processQueued();

        // ── Radial menu toggle and input ──
        if (radialMenuHandler) {
            radialMenuVisible = radialMenuHandler->update(loadedTextureLayers);
        }

        shadowParams.update(camera.getPosition(), light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);

        // Drain the pending mesh queue populated by the background scene-loading
        // thread.  GPU uploads happen here on the main thread so newly generated
        // chunks become visible progressively without blocking the render loop.
        // Process pending meshes at a controlled rate (10 per frame).
        // Chunks closest to the camera are uploaded first. Drains pending
        // entries from the ONE shared queue.
        if (sceneRenderer && !isLoading) {
            // Dynamic layers: pick up LayersWidget add/remove without restart.
            syncLayers();
            // Video-layer mesh uploads stage their indirect commands into the
            // renderer's per-frame cull slot, so the slot must already match the
            // frame being prepared (solid/water set theirs in preRenderPass; the
            // video pass only culls later, after the uploads).
            if (sceneRenderer->videoRenderer) {
                sceneRenderer->videoRenderer->getIndirectRenderer().setCullFrame(getCurrentFrame());
            }
            // Hybrid RT: water volumes join the proxy only while water renders.
            sceneRenderer->rtWaterProxyEnabled = settings.waterEnabled;
            std::deque<PendingMeshData> pendingBatch;
            sceneRenderer->drainPendingMeshes(pendingBatch, 16);
            sceneRenderer->processPendingMeshes(this, camera.getPosition(), pendingBatch);
        }

#ifdef DEBUG
        // VRAM headroom watchdog: on 4 GB iGPUs (e.g. Radeon 680M) exceeding
        // device-local memory makes radv/amdgpu cancel the CS -> device lost
        // (observed at the end of the bulk chunk-upload burst). Log the
        // device-local heap usage every ~5 s so regressions are visible in
        // run.log instead of surfacing only as a mysterious device lost.
        if (++vramWatchdogCounter_ >= 300) {
            vramWatchdogCounter_ = 0;
            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(getPhysicalDevice(), &memProps);
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
            vmaGetHeapBudgets(getVmaAllocator(), budgets);
            uint64_t usedMB = 0, totalMB = 0;
            for (uint32_t h = 0; h < memProps.memoryHeapCount; ++h) {
                if (memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    usedMB += budgets[h].usage;
                    totalMB += budgets[h].budget ? budgets[h].budget : memProps.memoryHeaps[h].size;
                }
            }
            std::cout << "[VRAM] device-local " << (usedMB >> 20) << " / "
                      << (totalMB >> 20) << " MB" << std::endl;
        }
#endif

        // Drive the async streaming subsystem each frame. prepareFrameWaits()
        // registers upload completion semaphores with this frame's submit, and
        // processUploads() submits as many queued jobs as staging allows — no
        // fixed per-frame cap. (Terrain/water copies currently still go
        // through IndirectRenderer; this is the integration point for migrating
        // them onto UploadManager.)
        if (sceneRenderer)
            sceneRenderer->streamer.update(this);

        // Synchronously complete the initial streaming uploads once so the base
        // terrain/water chunks are resident on the GPU before the first draw
        // (otherwise the indirect buffers are empty for the first frame and the
        // user sees a black screen). Subsequent streaming stays asynchronous.
        if (sceneRenderer) {
            static bool sFlushedInitial = false;
            if (!sFlushedInitial) {
                sceneRenderer->streamer.uploadManager().flush();
                sFlushedInitial = true;
            }
        }

        // Drain the vegetation placement queue so chunkBuffers is populated
        // before preRenderPass records read barriers.  Must happen here
        // because barriers cannot be emitted inside dynamic rendering, and
        // draw() runs inside beginPass/endPass.
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->processPendingChunks(10);
        }

        mainTime += deltaTime;
        lastFrameDelta = deltaTime;
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->setWindTime(mainTime);
            sceneRenderer->vegetationRenderer->setImpostorDistance(settings.impostorDistance);
        }
        if (sceneRenderer && sceneRenderer->skyRenderer) {
            sceneRenderer->skyRenderer->setCloudTime(mainTime);
            sceneRenderer->skyRenderer->setCloudsEnabled(settings.cloudsEnabled);
            // Ray-cast quality also drives the sky cloud raymarch (one cloud
            // ray per NxN screen-pixel block).
            sceneRenderer->skyRenderer->setRaycastPixelSize(settings.raycastPixelSize);
            sceneRenderer->skyRenderer->update(this);
        }

        // ── Brush undo (Ctrl+Z, Shift+Ctrl+Z for the <1 s group) ────────────
        pollBrushUndo();

        // ── Brush apply mode (Click vs Drag) ────────────────────────────────
        // In Drag mode, SPACE (keyboard) or B (Wiimote) applies continuously
        // every frame while held, enabling drag-apply across the terrain.
        // In Click mode, the edge-triggered event from the publisher handles
        // the single press — the continuous check below is skipped.
        {
            const ControllerParameters& cp = *controllerManager.getParameters();
            bool applyHeld = false;

            // Keyboard: SPACE held on BRUSH page in Drag mode
            bool spaceHeld = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
            if (spaceHeld
                && controllerManager.keyboardContext.activeCategory() == PageCategory::BRUSH
                && cp.keyboardBrushMode == BrushApplyMode::Drag)
            {
                applyHeld = true;
            }

            // Wiimote: B button held on BRUSH page in Drag mode
            // WIIMOTE_BUTTON_B = 0x0004 (defined in wiiuse.h)
            if (!applyHeld) {
                static constexpr uint16_t kWiimoteButtonB = 0x0004;
                WiimoteState wmState = nunchukPublisher.getState();
                bool bHeld = (wmState.buttons & kWiimoteButtonB) != 0;
                if (bHeld
                    && controllerManager.wiimoteContext.activeCategory() == PageCategory::BRUSH
                    && cp.wiimoteBrushMode == BrushApplyMode::Drag)
                {
                    applyHeld = true;
                }
            }

            if (applyHeld) {
                brushApplyToScenePending = true;
            }
        }

        profileCpuUpdate = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuUpdateT0).count();
    }

    // Begin a ONE_TIME_SUBMIT primary command buffer for an async pass.
    // `name` completes the error context after "vkBeginCommandBuffer failed
    // for"; `tag` selects the log prefix. On failure the buffer is freed and
    // VK_NULL_HANDLE is returned so the caller can decide how to bail out.
    VkCommandBuffer beginAsyncTask(const char* name, const char* tag = "[Async]") {
        VkCommandBuffer cmd = allocatePrimaryCommandBuffer();
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            std::cerr << tag << " vkBeginCommandBuffer failed for " << name << std::endl;
            freeCommandBuffer(cmd);
            return VK_NULL_HANDLE;
        }
        return cmd;
    }

    // Gather the scene views the water sets sample (solid color/depth,
    // vegetation color/depth) for `sceneFrameIdx` and bind them together with
    // the already-resolved back-face/RT/sky views into `ds`. The offscreen
    // water sets pass the current frame; water-in-main passes the previous
    // frame because the current main targets are the render attachments.
    void updateWaterSceneTextures(VkDescriptorSet ds, VkImageView backFaceDepth,
                                  VkImageView rtReflect, VkImageView rtRefract,
                                  VkImageView skyView, uint32_t sceneFrameIdx) {
        VkImageView solidColor = sceneRenderer->solidRenderer
            ? sceneRenderer->solidRenderer->getColorView(sceneFrameIdx) : VK_NULL_HANDLE;
        VkImageView solidDepth = sceneRenderer->solidRenderer
            ? sceneRenderer->solidRenderer->getDepthView(sceneFrameIdx) : VK_NULL_HANDLE;
        VkImageView vegColor = sceneRenderer->vegetationRenderer
            ? sceneRenderer->vegetationRenderer->getVegColorView(sceneFrameIdx) : VK_NULL_HANDLE;
        VkImageView vegDepth = sceneRenderer->vegetationRenderer
            ? sceneRenderer->vegetationRenderer->getVegDepthView(sceneFrameIdx) : VK_NULL_HANDLE;
        sceneRenderer->waterRenderer->updateSceneTexturesBinding(this, ds,
            backFaceDepth, rtReflect, rtRefract, skyView, solidColor, solidDepth, vegColor, vegDepth);
    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {

        // H9: a water render-scale change rebuilds the water-side offscreen
        // targets (color/body/column, geometry depth, back-face depth). They are
        // written on the water queue and read by the composite
        // on the graphics queue, so a graphics-queue-scoped wait would not cover
        // every consumer — this is the "major resource rebuild" case where
        // AGENTS.md allows a device idle. It runs once, on the frame the user
        // changes the slider.
        // L15: same idle path regrows/shrinks the body+column aux targets when
        // the blur/pipeline selection flipped since creation (layer edits,
        // preset switches). waterBodyTargetsStale() compares live need
        // against allocation; creation itself is a no-op when nothing changed.
        auto* liquidRenderer = sceneRenderer ? sceneRenderer->waterRenderer.get() : nullptr;
        if (sceneRenderer && (sceneRenderer->waterRenderScale() != settings.waterRenderScale ||
                              (liquidRenderer && liquidRenderer->waterBodyTargetsStale()))) {
            vkDeviceWaitIdle(getDevice());
            sceneRenderer->setWaterRenderScale(settings.waterRenderScale);
            sceneRenderer->recreateWaterTargets(this, getWidth(), getHeight());
        }
        // M12 (perf report 22): same idle path for a vegetation render-scale
        // change (the veg pass is written on the vegetation queue, so a
        // graphics-scoped wait would not cover it; this is a major rebuild).
        if (sceneRenderer && sceneRenderer->vegetationRenderScale() != settings.vegetationRenderScale) {
            vkDeviceWaitIdle(getDevice());
            sceneRenderer->setVegetationRenderScale(settings.vegetationRenderScale);
            sceneRenderer->recreateVegetationTargets(this, getWidth(), getHeight());
        }
        // C1 (perf report 23): a texture-array resolution change rebuilds the
        // five material arrays (device idle, full re-upload, mixer refresh).
        // Layer indices are resolution-independent, so every consumer binding
        // (materials, mixer targets, pickers) stays valid; this runs once, on
        // the frame the setting changes.
        {
            const uint32_t wantSize = static_cast<uint32_t>(std::clamp(settings.textureArraySize, 256, 4096));
            if (textureArrayManager.layerAmount != 0 && textureArrayManager.width != wantSize) {
                recreateTextureArrays(wantSize);
            }
        }

        uint32_t frameIdx = getCurrentFrame();

        // Per-frame cull buffers: each IndirectRenderer needs its own per-frame
        // compact/visibleCount buffer to avoid cross-frame overwrite races.
        // Must be set BEFORE prepareCull below so culls and drawPrepared use
        // the same per-frame compact/visibleCount slots (setCullFrame in
        // draw() would make every draw read a stale, never-culled slot).
        sceneRenderer->solidRenderer->getIndirectRenderer().setCullFrame(frameIdx);
        sceneRenderer->waterRenderer->getIndirectRenderer().setCullFrame(frameIdx);
        if (sceneRenderer->debugSDFRenderer) {
            sceneRenderer->debugSDFRenderer->setCullFrame(frameIdx);
            // The solid IndirectRenderer performs the SDF cube cull + compaction in
            // its OWN IndirectRenderer.comp dispatch (folded into the terrain cull), so point
            // the SDF debug renderer at it to draw from its SDF output buffers.
            sceneRenderer->debugSDFRenderer->setIndirectRenderer(&sceneRenderer->solidRenderer->getIndirectRenderer());
        }
        if (sceneRenderer->boundingBoxRenderer) {
            sceneRenderer->boundingBoxRenderer->setCullFrame(frameIdx);
            // Bounding-box frustum cull is folded into the solid IndirectRenderer's
            // IndirectRenderer.comp dispatch, so draw from its bbox output buffers.
            sceneRenderer->boundingBoxRenderer->setIndirectRenderer(&sceneRenderer->solidRenderer->getIndirectRenderer());
        }

        // Profiling: read previous frame's query results (with availability flag to
        // avoid even partial driver stalls), then reset for this frame.
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
            if (queryPoolReady[frameIdx]) {
                auto msDiff = [&](uint64_t endTs, uint64_t startTs) -> float {
                    return static_cast<float>(endTs - startTs) * timestampPeriod * 1e-6f;
                };
                // Group A: indices 6-9 (depth prepass, sky). Slots 0-5
                // (legacy shadow/cull/brush slots) are written by no pass.
                struct { uint64_t value; uint64_t availability; } tsA[4] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 6, 4,
                        sizeof(tsA), tsA, sizeof(tsA[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsA[0].availability) profileDepthPrepass = msDiff(tsA[1].value, tsA[0].value);
                    if (tsA[2].availability) profileSky          = msDiff(tsA[3].value, tsA[2].value);
                }
                // Group B: indices 10-19 (solid draw, vegetation, water, postprocess, imgui)
                struct { uint64_t value; uint64_t availability; } tsB[10] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 10, 10,
                        sizeof(tsB), tsB, sizeof(tsB[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsB[0].availability) profileSolidDraw          = msDiff(tsB[1].value, tsB[0].value);
                    if (tsB[2].availability) profileVegetationImpostor = msDiff(tsB[3].value, tsB[2].value);
                    if (tsB[4].availability) profileWater              = msDiff(tsB[5].value, tsB[4].value);
                    if (tsB[6].availability) profilePostProcess        = msDiff(tsB[7].value, tsB[6].value);
                    if (tsB[8].availability) profileImGui              = msDiff(tsB[9].value, tsB[8].value);
                }
                // Group C: indices 20-21 (hybrid-RT water dispatch, own CB slots)
                struct { uint64_t value; uint64_t availability; } tsC[2] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 20, 2,
                        sizeof(tsC), tsC, sizeof(tsC[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsC[0].availability) profileRTDispatch = msDiff(tsC[1].value, tsC[0].value);
                }
                // Group D: indices 22-23 (SDF engine ray-march pass, own CB slots)
                struct { uint64_t value; uint64_t availability; } tsD[2] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 22, 2,
                        sizeof(tsD), tsD, sizeof(tsD[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsD[0].availability) profileSdf = msDiff(tsD[1].value, tsD[0].value);
                }
                // SDF march counters (set=1 binding 10): snapshot the slot's
                // last completed values (3-frame latency) for the panel.
                if (sceneRenderer && sceneRenderer->sdfRenderer) {
                    sceneRenderer->sdfRenderer->readProfile(frameIdx);
                }
            }
            // Throttled console dump of the previous frame's GPU passes so slow
            // frames are attributable from run.log without the ImGui panel.
            {
                static uint32_t profilePrintTick = 0;
                if ((++profilePrintTick & 0x1F) == 0) {
                    const float gpuTotal = profileDepthPrepass + profileSky +
                        profileSolidDraw + profileVegetationImpostor + profileWater +
                        profileRTDispatch + profileSdf + profilePostProcess + profileImGui;
                    if (gpuTotal > 40.0f) {
                        std::cout << "[gpu] total=" << gpuTotal
                                  << " depth=" << profileDepthPrepass
                                  << " sky=" << profileSky
                                  << " solid=" << profileSolidDraw
                                  << " veg=" << profileVegetationImpostor
                                  << " water=" << profileWater
                                  << " rt=" << profileRTDispatch
                                  << " sdf=" << profileSdf
                                  << " post=" << profilePostProcess
                                  << " imgui=" << profileImGui
                                  << " fps=" << profileFps << std::endl;
                    }
                }
            }
            // Reset only the slots owned by THIS command buffer. The solid pass
            // records on its own command buffer (solidQueue) and resets its own
            // slots 6-11 there; the vegetation pass owns 12-13 (vegQueue), the
            // water pass 14-15 and the SDF pass 22-23 (their own CBs); the
            // postprocess/ImGui passes own 16-19. Splitting the reset across
            // command buffers avoids a cross-queue reset/write race. Slots 0-5
            // (shadow/cull/brush) are written by no pass at all, so they are no
            // longer reset: an unwritten slot already reads back unavailable and
            // the readback skips it.
            vkCmdResetQueryPool(commandBuffer, queryPools[frameIdx], 16, 4);  // 16-19: postprocess/imgui
            queryPoolReady[frameIdx] = true;
        }

        // Per-op RT profiling: bind the instrumented (RT_PROFILE) pipeline
        // variants while the toggle is on, and read + reset this slot's counter
        // block. The slot was last written MAX_FRAMES_IN_FLIGHT frames ago and
        // its fence has signaled (per-slot wait in drawFrame), so the read and
        // the CPU memset never race the GPU's atomics. Requires
        // VK_KHR_shader_clock (the profile variants are only built with it).
        if (sceneRenderer && sceneRenderer->rayTracing) {
            const bool rtProf = rtProfilingEnabled_ && rtProfilingSupported
                && sceneRenderer->rayTracing->isSupported();
            if (sceneRenderer->solidRenderer)
                sceneRenderer->solidRenderer->setRtProfilingEnabled(rtProf);
            if (sceneRenderer->waterRenderer)
                sceneRenderer->waterRenderer->setRtProfilingEnabled(rtProf);
            if (rtProf) {
                sceneRenderer->rayTracing->readProfile(frameIdx, rtProfileStats_);
                sceneRenderer->rayTracing->resetProfile(frameIdx);
            }
        }

        auto cpuRecordT0 = std::chrono::high_resolution_clock::now();

        // Rebuild projection matrix FIRST so viewProj below reflects current near/far settings
        {
            float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
            glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, settings.nearPlane, settings.farPlane);
            proj[1][1] *= -1; // Vulkan Y-flip
            camera.setProjection(proj);
        }

        // Build per-frame UBO (viewProj now includes the updated projection)
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        uboStatic.viewProjection = viewProj;
        uboStatic.invViewProjection = glm::inverse(viewProj);
        uboStatic.viewPosition = camera.getPosition();
        uboStatic.lightDirection = glm::vec3(light.getDirection());
        uboStatic.lightElevation = uboStatic.lightDirection.y;
        uboStatic.lightColor = glm::vec3(1.0f, 1.0f, 0.9f);
        uboStatic.lightSpaceMatrix  = shadowParams.lightSpaceMatrix[0];
        uboStatic.lightSpaceMatrix1 = shadowParams.lightSpaceMatrix[1];
        uboStatic.lightSpaceMatrix2 = shadowParams.lightSpaceMatrix[2];
        // Canonical scene UBO fields (see vulkan/ubo/UniformObject.hpp)
        uboStatic.debugMode = settings.debugMode;
        uboStatic.roughnessEnabled = settings.roughnessEnabled ? 1u : 0u;
        uboStatic.ambientOcclusionEnabled = settings.aoEnabled ? 1u : 0u;
        uboStatic.triplanarThreshold = settings.triplanarThreshold;
        uboStatic.triplanarExponent = settings.triplanarExponent;
        uboStatic.tessNearDist = settings.tessMinDistance;
        uboStatic.tessFarDist = settings.tessMaxDistance;
        uboStatic.tessellationFactor = settings.tessellationFactor;
        uboStatic.isShadowPass = 0u;
        uboStatic.tessellationEnabled = settings.tessellationEnabled ? 1u : 0u;
        uboStatic.nearPlane = settings.nearPlane;
        uboStatic.farPlane = settings.farPlane;
        uboStatic.normalMappingEnabled = settings.normalMappingEnabled ? 1u : 0u;
        uboStatic.shadowsEnabled = settings.enableShadows ? 1u : 0u;

        // Brush params: brushTextureIndex, brushMode, brushHSV
        {
            uint32_t brushTexIdx = 0u;
            uint32_t brushMode = 0u;
            glm::vec3 brushHSV(0.0f, 0.5f, 0.5f);
            const BrushEntry* brushEntry = brushManager.getSelectedEntry();
            if (brushEntry) {
                brushTexIdx = static_cast<uint32_t>(brushEntry->materialIndex);
                brushMode = static_cast<uint32_t>(brushEntry->brushMode);
                brushHSV = brushEntry->hsv;
            }
            uboStatic.brushTextureIndex = brushTexIdx;
            uboStatic.brushMode = brushMode;
            uboStatic.brushPhase = static_cast<float>(glfwGetTime());
            uboStatic.brushHsv = brushHSV;
        }

        // Hybrid RT per-frame params (contents stream; handles stable). The
        // TLAS-readiness flag (debug.y) flips once the first BLAS/TLAS build is
        // recorded — GPU ordering (samplers wait tlCull, builds precede its
        // signal) makes same-frame sampling safe.
        if (sceneRenderer && sceneRenderer->rayTracing) {
            // Pipeline-path water look mirrors water layer 0 (rgen has no
            // layer id); the sampled inline path reads per-fragment layers.
            static const WaterSettings kDefaultWaterLook{};
            const WaterSettings& waterLook = waterParams.empty() ? kDefaultWaterLook : waterParams[0];
            sceneRenderer->updateRTParams(this, settings, waterLook, uboStatic.invViewProjection,
                uboStatic.viewPosition, -uboStatic.lightDirection,
                uboStatic.lightColor, settings.nearPlane, settings.farPlane);
        }

        // Reset command buffer state tracker and wire it to all sub-renderers.
        // NOTE: backFaceRenderer and the water IndirectRenderer are deliberately
        // excluded by SceneRenderer::setCmdState — they are accessed by the
        // async back-face task on a separate thread and keeping cmdState=nullptr
        // for them avoids a data race on frameCmdState.
        sceneRenderer->frameCmdState.reset();
        sceneRenderer->setCmdState(&sceneRenderer->frameCmdState);

        // ── GPU culling (solid / vegetation / brush / water) and the early brush
        // pass now run on a SEPARATE async command buffer (the "cull" task) that
        // signals semMainCull. This keeps the heavy solid/vegetation shading on
        // this main command buffer free to overlap the cull + shadow +
        // water + vegetation offscreen work. The shadow pass, the water geometry
        // pass, and the vegetation pass each run on their own async command buffer
        // and wait on semMainCull (directly or via queue ordering) so the shared
        // visibleLods / brush-depth buffers are current before they read them.

        // NOTE: the shadow pass has been moved to its own async command buffer
        // (see the "shadow" task launched later in draw()). It signals semShadow,
        // which the main command buffer waits on below so the shadow map and the
        // restored UBO / visibleLods are ready before the solid shading samples
        // them.

        // Upload UBO to GPU (VMA persistently mapped — no map/unmap needed)
        // Upload UBO to GPU (VMA persistently mapped — no map/unmap needed).
        // When shadows are ENABLED the shadow pass (now its own async command
        // buffer) restores this buffer to uboStatic at the end of its render, so
        // we must NOT also write it here (that would be a concurrent host/GPU
        // write on the same buffer). When shadows are DISABLED the shadow task
        // early-returns without restoring, so the main CB must upload uboStatic.
        if (sceneRenderer) {
            if (!settings.enableShadows) {
                memcpy(sceneRenderer->mainUniformBuffers[frameIdx].mappedData, &uboStatic, sizeof(UniformObject));
            }
        } else {
            std::cerr << "[MyApp::preRenderPass] sceneRenderer is null, skipping UBO upload\n";
        }

        // Grass mode selector (Settings widget): bladed grass (triangle tufts
        // + captured impostors) or none. vegetationEnabled derives from it.
        if (settings.grassMode != appliedGrassMode) {
            appliedGrassMode = settings.grassMode;
            settings.vegetationEnabled =
                (settings.grassMode == Settings::GrassMode::Bladed);
        }
        if (sceneRenderer && sceneRenderer->sdfRenderer) {
            // Ray-cast quality (1..8 px blocks; default 2) for the SDF pass.
            sceneRenderer->sdfRenderer->setRaycastPixelSize(settings.raycastPixelSize);
            // March tiers (perf report 25 M12): step budget + smoke resolve
            // samples. The setters dedupe unchanged values (params slots go
            // dirty only on change), so these per-frame calls are cheap; the
            // values stream through SdfParamsUBO with no idle and no target
            // rebuild (unlike the water/vegetation render-scale path above).
            // No idle-guarded recreate applies here by design.
            sceneRenderer->sdfRenderer->setMaxSteps(settings.sdfMaxSteps);
            sceneRenderer->sdfRenderer->setSmokeSamples(settings.sdfSmokeSamples);
            // Ray-marching master gate (Settings toggle / Minimal preset):
            // off clears the SDF targets and skips the grass shadow caster.
            sceneRenderer->sdfRenderer->setRayMarchingEnabled(settings.rayMarchingEnabled);
            // M10 (report 25): current view-projection for the CPU container
            // screen-overlap gate on the solid-depth preload copy.
            sceneRenderer->sdfRenderer->setOcclusionViewProj(camera.getViewProjectionMatrix());
        }

        const bool waterEnabled = settings.waterEnabled;
        const bool vegetationEnabled = settings.vegetationEnabled;
        // Bladed-grass impostor hand-off: captured blade tufts take over beyond
        // Settings::impostorDistance, so the impostor pass runs whenever
        // vegetation is enabled and the hand-off distance is set.
        const bool grassImpostorsEnabled =
            settings.vegetationEnabled && settings.impostorDistance > 0.0f;

        // ── Solid scene pass (sky offscreen + depth prepass + color pass) ──
        // Recorded on its OWN command buffer (solidCmd) and submitted to the
        // dedicated solidQueue. It waits on the cull task's visibleLods/brush depth
        // (semCullSolid) and the shadow map + restored UBO (semShadowSolid), then
        // signals semSolid. The main command buffer (postprocess), the water task
        // composite waits on tlSolid transitively. The "solid" task is launched
        // later in draw().

        // ── Early brush pass moved to its own async command buffer (see the
        // "cull" task launched later in draw()). The brush pass writes brush
        // depth. (The legacy solid360 cubemap task was removed — water samples
        // the RT outputs + sky now.)

        // (Instance 1 depth pre-pass moved to the dedicated solid command buffer.)

        // (Instance 2 color pass moved to the dedicated solid command buffer.)


        // If water is disabled, clear its offscreen targets here (outside any active
        // dynamic rendering instance) so the post-process compositor won't sample
        // stale content.
        if (!waterEnabled && sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->clearRenderTargets(this, commandBuffer, frameIdx);
        }

        // Launch asynchronous recording+submit for independent offscreen passes
        // using a persistent thread pool to avoid per-frame std::thread creation overhead.
        // Dependency graph (timeline semaphores, one per producer):
        //   cull task      : brush early pass + all GPU culls -> tlCull
        //                    (the throttled RT BLAS/TLAS rebuild rides its own
        //                    command buffer on the solid queue, ahead of the
        //                    solid pass — see the cull task)
        //   shadow task    : shadow map + cascade cull + blur, restores visibleLods/UBO
        //                    (runs after cull on the worker) -> tlShadow
        //                    (not enqueued when shadows are disabled; consumers
        //                    then wait tlCull@v directly)
        //   backface+water : back-face depth pass + water geometry pass + RT dispatch
        //                    (waits tlSolid + tlSky) -> tlWater, or a clear-only
        //                    fast path when the previous cull saw no water
        //   vegetation task: own offscreen color+depth (runs after shadow on the
        //                    worker, so the shadow map is current) -> tlVeg
        //   sdf task       : own offscreen color+depth (debug SDF cubes) -> tlSdf
        //   bbox task       : own offscreen color+depth (mesh bounding boxes) -> tlBbox
        // The main command buffer waits on the producer timelines via
        // m_extraWaitSemaphores / m_compositeTimelineWaits, which the tasks
        // register their signal semaphores into.
        // Frame-graph timeline semaphores: one per producer. A timeline semaphore's
        // counter is monotonic, so a single semaphore per producer is waited by every
        // consumer of that frame — collapsing the many binary per-consumer semaphores
        // into one. Created once and reused across frames; per-frame ordering is the
        // monotonically increasing tlFrameValue. They are NOT destroyed per frame
        // (unlike the old binary semaphores) — ResourceManager owns them until teardown.
        static VkSemaphore tlCull = VK_NULL_HANDLE, tlShadow = VK_NULL_HANDLE, tlSky = VK_NULL_HANDLE;
        static VkSemaphore tlSolid = VK_NULL_HANDLE, tlBrush = VK_NULL_HANDLE;
        static VkSemaphore tlVeg = VK_NULL_HANDLE, tlSdf = VK_NULL_HANDLE, tlBbox = VK_NULL_HANDLE;
        static VkSemaphore tlWater = VK_NULL_HANDLE;
        static uint64_t tlFrameValue = 0;
        // Last frame value signaled on tlWater (0 = none yet). The async RT
        // build waits on it so it cannot overwrite the TLAS/metas while a
        // previous frame's water pass (different queue) is still reading them.
        static uint64_t lastWaterSignal = 0;
        // Last frame value signaled on tlVeg (0 = none yet). The vegetation
        // consolidation copy waits on it so its overwrite of the inactive
        // instance pair never races the previous era's vegetation draws
        // (vegetation queue) or the shadow cascade draws (cube queues,
        // transitively covered because the veg task waits tlShadow first).
        // Flicker fix: without this wait the copy overwrote instance bytes
        // under in-flight draws and every streaming consolidation flashed
        // wrong instances for a frame.
        static uint64_t lastVegSignal = 0;
        static bool tlInit = false;
        if (!tlInit) {
            tlCull      = createTimelineSemaphore();
            tlShadow    = createTimelineSemaphore();
            tlSky       = createTimelineSemaphore();
            tlSolid     = createTimelineSemaphore();
            tlBrush     = createTimelineSemaphore();
            tlVeg       = createTimelineSemaphore();
            tlSdf       = createTimelineSemaphore();
            tlBbox      = createTimelineSemaphore();
            tlWater     = createTimelineSemaphore();
            tlInit = true;
        }
        const uint64_t v = ++tlFrameValue;
        std::future<void> asyncCullFuture, asyncBackFaceFuture;
        std::future<void> asyncShadowFuture, asyncVegFuture, asyncSdfFuture, asyncBboxFuture;
        std::future<void> asyncSolidFuture, asyncSkyFuture;

        // --- Cull + early brush pass on its own command buffer (signals semMainCull) ---
        if (sceneRenderer) {
            // Hand the vegetation renderer the cross-queue wait for its
            // consolidation copy BEFORE the cull task may submit one
            // (flicker fix: see lastVegSignal above).
            if (sceneRenderer->vegetationRenderer)
                sceneRenderer->vegetationRenderer->setConsolidationWait(tlVeg, lastVegSignal);
            asyncCullFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, v, grassImpostorsEnabled]() {
                MyApp* app = this;
                VkCommandBuffer cullCmd = app->beginAsyncTask("cull");
                if (cullCmd == VK_NULL_HANDLE) return;
                // Timeline semaphore: the cull results are signaled once on tlCull with
                // the current frame value v. Every consumer (shadow, veg, sdf, bbox,
                // solid, brush-solid, water, solid360, and the composite) waits on
                // tlCull@v, which is valid because a timeline semaphore may have many
                // waiters on the same value (unlike a binary semaphore).
                // Use a dedicated command-buffer state for this async command buffer.
                // Sharing frameCmdState with the main CB would let the cached
                // "last bound pipeline" from a different command buffer suppress the
                // vkCmdBindPipeline before a dispatch (the validation error we hit).
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);
                this->sceneRenderer->solidRenderer->getIndirectRenderer().acquireBuffers(cullCmd);
                if (this->sceneRenderer->vegetationRenderer &&
                    (settings.vegetationEnabled || grassImpostorsEnabled))
                    this->sceneRenderer->vegetationRenderer->prepareCull(cullCmd, viewProj);
                if (settings.showSDFDebug && this->sceneRenderer && this->sceneRenderer->debugSDFRenderer)
                    this->sceneRenderer->debugSDFRenderer->registerToIndirect();
                if (settings.showBoundingBoxes && this->sceneRenderer && this->sceneRenderer->boundingBoxRenderer)
                    this->sceneRenderer->boundingBoxRenderer->registerBoundingBoxesToIndirect();
                else if (this->sceneRenderer && this->sceneRenderer->boundingBoxRenderer)
                    this->sceneRenderer->boundingBoxRenderer->clearBoundingBoxesToIndirect();
                this->sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(cullCmd, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
                if (settings.waterEnabled && this->sceneRenderer->waterRenderer) {
                    // Ensure water's pending uploads (vertex/index) have completed and their
                    // deferred meta (indirect/bounds) is visible before we cull.
                    this->sceneRenderer->waterRenderer->getIndirectRenderer().pollPendingTransfers(this);
                    this->sceneRenderer->waterRenderer->getIndirectRenderer().syncHostBuffersToGPU();
                    this->sceneRenderer->waterRenderer->getIndirectRenderer().prepareCull(cullCmd, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod, nullptr, false, true, false, 0, 1);
                }
                if (settings.showSDFDebug && this->sceneRenderer && this->sceneRenderer->debugSDFRenderer)
                    this->sceneRenderer->debugSDFRenderer->prepareCull(cullCmd);
                // Hybrid RT: (re)build the stable proxy BLAS/TLAS when chunk
                // changes staged new boxes. Throttled inside (<=1/30 frames);
                // camera/LOD/tessellation never mark dirty (§6). The build is
                // recorded on its OWN command buffer and submitted to the solid
                // queue BEFORE the solid pass, so it no longer serializes with
                // the graphics-queue cull/main work. Same-queue FIFO orders it
                // ahead of the solid pass that consumes the new TLAS, and the
                // water pass waits tlSolid@v (signaled after the build).
                // Previous frames' water passes (different queue) may still
                // read the metas/TLAS, so the build waits on tlWater's last
                // signaled value; previous solid passes are same-queue FIFO.
                if (this->sceneRenderer && this->sceneRenderer->rayTracing &&
                    this->sceneRenderer->rayTracing->wantsBuild()) {
                    VkCommandBuffer buildCmd = app->allocatePrimaryCommandBuffer();
                    VkCommandBufferBeginInfo bbegin{};
                    bbegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    bbegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    if (vkBeginCommandBuffer(buildCmd, &bbegin) == VK_SUCCESS &&
                        this->sceneRenderer->rayTracing->buildIfNeeded(this, buildCmd)) {
                        std::vector<VkSemaphore> buildWaits;
                        std::vector<uint64_t> buildWaitValues;
                        if (lastWaterSignal > 0) {
                            buildWaits.push_back(tlWater);
                            buildWaitValues.push_back(lastWaterSignal);
                        }
                        app->submitCommandBufferAsyncToQueue(buildCmd, app->getSolidQueue(),
                            nullptr, buildWaits, false, {}, buildWaitValues, 0, {}, false);
                    } else {
                        app->freeCommandBuffer(buildCmd);
                    }
                }
                // Signal the single cull timeline semaphore; consumers wait on tlCull@v.
                // Cull is the root producer. The composite no longer waits tlCull
                // directly (it is transitively implied through tlSolid/tlWater and
                // the brush preview's own registered signal), so it is not
                // registered into the composite's wait list. Every consumer that
                // needs cull results waits it explicitly or transitively.
                app->submitCommandBufferAsyncToQueue(cullCmd, app->getGraphicsQueue(), &tlCull, {}, false, {}, {}, v, {}, false);
            });
            asyncCullFuture.get();
            // Restore the per-frame state for the main CB's continued recording.
            this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);

            // --- Brush preview offscreen on its own queue (signals tlBrush) ---
            // SDF-raymarched brush preview (color + front depth) into the brush
            // targets the composite overlays. No tessellated brush geometry and
            // no dependency on the solid pass: the solid shader evaluates the
            // same brush params (set=1) directly for PAINT/REMOVE. The signal is
            // registered for the composite, which samples the brush targets.
            {
                updateBrushPreview(frameIdx);
                VkCommandBuffer brushCmd = beginAsyncTask("brush preview", "[MyApp]");
                if (brushCmd == VK_NULL_HANDLE) {
                    // First begin failed (CB allocation hiccup): retry the
                    // allocation so tlBrush@v is still signaled — the
                    // composite's registered timeline wait would otherwise
                    // deadlock.
                    brushCmd = allocatePrimaryCommandBuffer();
                    VkCommandBufferBeginInfo bi{};
                    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    if (brushCmd == VK_NULL_HANDLE || vkBeginCommandBuffer(brushCmd, &bi) != VK_SUCCESS) {
                        if (brushCmd != VK_NULL_HANDLE) freeCommandBuffer(brushCmd);
                        brushCmd = VK_NULL_HANDLE;
                    }
                }
                if (brushCmd != VK_NULL_HANDLE) {
                    CommandBufferState brushState;
                    this->sceneRenderer->setCmdState(&brushState);
                    this->sceneRenderer->brushRenderer->recordPass(this, brushCmd, frameIdx, getMainDescriptorSet(),
                        this->sceneRenderer->mainUniformBuffers[frameIdx].buffer);
                    this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);
                    submitCommandBufferAsyncToQueue(brushCmd, getBrushQueue(), &tlBrush, {tlCull}, true, {}, {v}, v, {}, true);
                }
            }
        }


        // SDF frame inputs + depth bindings, set on the MAIN THREAD before the
        // shadow task records: the grass shadow pass binds the SDF descriptor
        // set, and vkUpdateDescriptorSets on a set referenced by a pending
        // command buffer is illegal (VUID 03047). refreshFrameBindings is
        // gated once per frame; the later SDF task only records barriers.
        if (sceneRenderer && sceneRenderer->sdfRenderer) {
            if (sceneRenderer->solidRenderer) {
                VkImageView dv = sceneRenderer->solidRenderer->getDepthView(frameIdx);
                if (dv != VK_NULL_HANDLE) {
                    // Image + extent too: render() pre-loads the SDF depth
                    // attachment with this solid depth so the hardware depth
                    // test rejects occluded SDF fragments at raster time.
                    sceneRenderer->sdfRenderer->setSceneDepth(dv,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        sceneRenderer->solidRenderer->getDepthImage(frameIdx),
                        sceneRenderer->solidRenderer->getRenderWidth(),
                        sceneRenderer->solidRenderer->getRenderHeight());
                }
            }
            if (sceneRenderer->waterRenderer) {
                const bool waterDepthValid = waterEnabled && !settings.waterInMainPass &&
                    sceneRenderer->waterRenderer->waterBlurNeeded();
                VkImageView wdv = sceneRenderer->waterRenderer->getWaterGeomDepthView(frameIdx);
                // A null view clears the enabled flag (binding keeps its
                // placeholder); refreshDepthBinding skips nulls.
                sceneRenderer->sdfRenderer->setWaterDepth(wdv,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    waterDepthValid && wdv != VK_NULL_HANDLE);
            }
            sceneRenderer->sdfRenderer->setFrame(frameIdx);
            // SDF march counters (profiling UI): re-applied per frame so a
            // renderer rebuild cannot leave a stale gate.
            sceneRenderer->sdfRenderer->setProfileCounters(
                sdfProfilingEnabled_ && sceneRenderer->sdfRenderer->profileCountersSupported());
            sceneRenderer->sdfRenderer->refreshFrameBindings(frameIdx);
        }

        // --- Shadow pass on its own command buffer (signals semShadow) ---
        // The shadow pass does its own cascade culling, blurs the cascades, and at
        // the end restores the shared visibleLods + main UBO (so the main command
        // buffer's solid shading reads the correct state). It depends on the cull
        // task's visibleLods (run earlier on the worker) and restores them, so the
        // main CB must wait on semShadow before sampling the shadow map / restored
        // UBO / visibleLods. It runs on the same graphics queue, after the cull
        // task (worker ordering), so the visibleLods are current when it reads them.
        // Shadows disabled: no task is enqueued (renderParallel would only submit
        // an empty CB relaying tlCull -> tlShadow). Nothing signals tlShadow on
        // those frames, so its consumers wait tlCull@v directly instead and no
        // worker is joined.
        {
            const bool shEnableShadows = settings.enableShadows;
            const bool shRenderSolid = settings.renderSolid;
            const bool shVegEnabled = vegetationEnabled;
            const bool shShadowTess = settings.shadowTessellationEnabled;
            const float shLodBias = settings.lodBias;
            const float shMaxTargetLod = settings.maxTargetLod;
            const glm::vec3 shCamPos = camera.getPosition();
            if (shEnableShadows) {
                asyncShadowFuture = asyncThreadPool.enqueue([this, frameIdx, viewProj, shEnableShadows, shRenderSolid, shVegEnabled, shShadowTess, shLodBias, shMaxTargetLod, shCamPos, v]() {
                    // The shadow producer signals a single timeline semaphore tlShadow@v.
                    // The shadow producer signals a single timeline semaphore tlShadow@v.
                    // Its internal cascade/blur sub-passes use their own (persistent binary)
                    // semaphores; only the cross-queue edge becomes a timeline wait. The
                    // composite does NOT wait on tlShadow (it only samples the final images,
                    // not the shadow map), so registerSignal is false; solid/veg/water/
                    // solid360 wait tlShadow@v via their own wait lists (tlCull@v when
                    // shadows are disabled and tlShadow is never signaled).
                    // Render each cascade on its own command buffer (parallel on distinct
                    // cube queues); the EVSM blur + main-camera cull restore run serially in
                    // a final CB that raises tlShadow once the shadow map is ready. Waits on
                    // tlCull@v so the cull GPU buffers are visible before the cascade draws
                    // read them.
                    this->sceneRenderer->shadowMapper->renderParallel(this, frameIdx,
                        sceneRenderer->mainUniformBuffers[frameIdx], uboStatic,
                        shEnableShadows, shRenderSolid, shVegEnabled, shShadowTess, shLodBias, shCamPos, shMaxTargetLod,
                        tlCull, v, tlShadow, v);
                });
            }
        }

        // --- Solid scene pass on its OWN command buffer (submitted to solidQueue) ---
        // Records the sky offscreen + solid depth pre-pass + solid color pass and
        // transitions the solid color/depth images to SHADER_READ_ONLY_OPTIMAL, all
        // on a dedicated command buffer that runs on solidQueue. It waits on the cull
        // task's visibleLods / brush depth (semCullSolid) and the shadow map + restored
        // UBO (semShadowSolid), then signals semSolid (consumed by the main composite
        // command buffer), semSolidWater (water task) and semSolid360 (solid360 task) —
        // three DISTINCT binary semaphores so none is waited twice. Submitting to
        // solidQueue (which may alias graphicsQueue on HW that exposes few graphics
        // queues) keeps the solid shading decoupled from the composite/postprocess work.
        // The solid task waits on the shadow task's timeline signal (on
        // graphicsQueue) when shadows are enabled. Join the shadow task first so
        // its signal submit is guaranteed to precede the solid task's wait submit
        // (binary semaphores can't be waited before their signal has been submitted
        // for execution). Shadows disabled: no shadow task exists, so this join is
        // skipped and the solid task waits tlCull@v directly.
        if (asyncShadowFuture.valid())
            asyncShadowFuture.get();

        // --- Sky offscreen on its OWN queue (skyQueue), in parallel with the solid
        //     pass. Writes the equirectangular sky image (sampled by water, solid360
        //     and the main composite for reflections). Signals semSky; the main
        //     composite auto-waits it (registered signal) and water/solid360 wait it
        //     explicitly. The fullscreen sky draw stays in the solid color pass.
        //
        // H8 (perf report 22): the cloudless equirect is view-independent —
        // SkyRendererEquirect.frag reads only the SkyUniform block and the light
        // direction/elevation, never the camera — so it is cached and
        // re-rendered only when those inputs change.
        // With clouds ON the equirect is camera- AND time-dependent by design:
        // SkyRendererEquirect.frag marches from ubo.viewPosition with the continuous
        // sky.cloudTime, exactly like the on-screen sky. Caching it (even at
        // the old 2 Hz quantization) freezes reflections in 0.5 s steps while
        // the sky moves smoothly, and a fixed origin would shift the
        // world-anchored pattern by the camera offset. So while clouds are
        // active the probe re-renders every frame; when off, the H8 cache
        // applies unchanged (cloudT collapses to 0).
        {
            SkySettings::Mode skyMode = this->sceneRenderer->getSkySettings().mode;
            const SkySettings& sky = this->sceneRenderer->getSkySettings();
            const CloudSettings& clouds = this->sceneRenderer->getCloudSettings();
            const glm::vec3 skyLightDir = glm::normalize(light.getDirection());
            const bool cloudsActive = settings.cloudsEnabled && clouds.enabled;
            // Quantize cloud time: 0.5 s steps balance drift smoothness vs cache hits.
            const float cloudT = settings.cloudsEnabled && clouds.enabled
                ? std::floor(mainTime * clouds.timeScale * 2.0f) / 2.0f : 0.0f;
            const float skySigNow[42] = {
                sky.horizonColor.x, sky.horizonColor.y, sky.horizonColor.z,
                sky.zenithColor.x, sky.zenithColor.y, sky.zenithColor.z,
                sky.warmth, sky.exponent, sky.sunFlare,
                skyLightDir.x, skyLightDir.y, skyLightDir.z,
                settings.cloudsEnabled && clouds.enabled ? 1.0f : 0.0f,
                clouds.lowEnabled ? 1.0f : 0.0f, clouds.midEnabled ? 1.0f : 0.0f, clouds.highEnabled ? 1.0f : 0.0f,
                clouds.lowCoverage, clouds.midCoverage, clouds.highCoverage, cloudT,
                clouds.lowDensity, clouds.midDensity, clouds.highDensity,
                clouds.densityScale, clouds.windResponse, 0.0f,
                clouds.lowScale, clouds.midScale, clouds.highScale,
                clouds.lowBaseHeight, clouds.midBaseHeight, clouds.highBaseHeight,
                clouds.lowThickness, clouds.midThickness, clouds.highThickness,
                clouds.detailStrength, clouds.exposure,
                clouds.silverLining, clouds.ambientBoost, clouds.sunForwardG,
                static_cast<float>(clouds.raymarchSteps), static_cast<float>(clouds.lightSteps)
            };
            bool skySigChanged = false;
            uint32_t skyRuns = 0;
            {
                // Written by the sky worker on successful begin (below).
                std::lock_guard<std::mutex> lock(skySigMutex);
                skySigChanged = !skySigValid;
                if (!skySigChanged) {
                    for (int i = 0; i < 42; ++i) {
                        if (skySigNow[i] != skySig[i]) { skySigChanged = true; break; }
                    }
                }
                skyRuns = skyRenderRuns;
            }
            // Clouds force a re-render every frame (see comment above):
            // continuous time + camera-anchored origin. Cloudless skies keep
            // the H8 cache (signature check only).
            const bool skyNeedsRender = !this->sceneRenderer->skyRenderer
                || skyRuns < SKY_CACHE_WARMUP || skySigChanged || cloudsActive;
            if (skyNeedsRender) {
                // NOTE: the signature commits only inside the task, after
                // beginAsyncTask succeeds. Committing it here would mark a
                // skipped/failed task as rendered, freezing a stale image
                // (e.g. a cloudy equirect persisting in water reflections
                // after disabling clouds) until the next parameter change.
                // A failed begin leaves the signature invalid, so the next
                // frame retries instead of caching stale contents.
                std::array<float, 42> sigCopy{};
                for (int i = 0; i < 42; ++i) sigCopy[i] = skySigNow[i];
                asyncSkyFuture = asyncThreadPool.enqueue([this, frameIdx, viewProj, skyMode, v, sigCopy]() {
                    MyApp* app = this;
                    VkCommandBuffer skyCmd = app->beginAsyncTask("sky");
                    if (skyCmd == VK_NULL_HANDLE) return;
                    {
                        std::lock_guard<std::mutex> lock(skySigMutex);
                        for (int i = 0; i < 42; ++i) skySig[i] = sigCopy[i];
                        skySigValid = true;
                        ++skyRenderRuns;
                    }
                    CommandBufferState taskState;
                    this->sceneRenderer->setCmdState(&taskState);
                    if (this->sceneRenderer->skyRenderer)
                        this->sceneRenderer->skyRenderer->renderOffscreen(this, skyCmd, frameIdx,
                            getMainDescriptorSet(), this->sceneRenderer->mainUniformBuffers[frameIdx], this->uboStatic, viewProj, skyMode);
                    this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);
                    // submitCommandBufferAsyncToQueue ends the command buffer and submits
                    // it (do NOT call vkEndCommandBuffer here). registerSignal=true ->
                    // tlSky is auto-waited by the main composite; tlSky@v is also waited by
                    // the water/back-face pass (a timeline semaphore allows multiple waiters).
                    app->submitCommandBufferAsyncToQueue(skyCmd, app->getSkyQueue(), &tlSky, {},
                        true, {}, {}, v, {}, true);
                });
            } else {
                // Cached frame: no equirect render, but the timeline still needs
                // GPU signal v for the composite/water/back-face waits. A HOST
                // signal is invalid here: VUID-VkSemaphoreSignalInfo-value-03259
                // requires the host value to be less than any pending GPU signal
                // value, and the previous frame's sky submission (v-1) can still
                // be in flight. Submit an empty command buffer whose only job is
                // the signal, carrying the normal registration so the frame
                // graph's wait set is identical to a rendered frame. The cost is
                // the submission itself -- the ~2M pixels of equirect shading and
                // the descriptor/layout churn are what the cache removes.
                VkCommandBuffer skyCmd = allocatePrimaryCommandBuffer();
                VkCommandBufferBeginInfo cbegin{};
                cbegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                cbegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(skyCmd, &cbegin) == VK_SUCCESS) {
                    asyncSkyFuture = asyncThreadPool.enqueue([this, skyCmd, v]() {
                        this->submitCommandBufferAsyncToQueue(skyCmd, this->getSkyQueue(), &tlSky, {},
                            true, {}, {}, v, {}, true);
                    });
                } else {
                    std::cerr << "[Sky] cached frame: vkBeginCommandBuffer failed" << std::endl;
                    freeCommandBuffer(skyCmd);
                }
            }
        }

        if (sceneRenderer && sceneRenderer->solidRenderer) {
            asyncSolidFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, v]() {
                MyApp* app = this;
                VkCommandBuffer solidCmd = app->beginAsyncTask("solid");
                if (solidCmd == VK_NULL_HANDLE) return;
                // Own command-buffer state (see cull task).
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);

                // Reset the query slots owned by this command buffer (depth
                // prepass, sky, solid draw) so the GPU profiling timestamps
                // below start from a clean state. Slots 12-13 belong to the
                // vegetation CB and 14-15/20-23 to the water/SDF CBs; the main
                // CB resets 16-19.
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                    vkCmdResetQueryPool(solidCmd, queryPools[frameIdx], 6, 6);

                // ── Sky offscreen moved to its own queue (asyncSkyFuture) so the
                //    equirectangular sky image is produced in parallel with the solid
                //    and shadow passes. The fullscreen sky (render()) still draws into
                //    this solid color pass below. ──

                VkClearValue colorClear{};
                colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                VkClearValue depthClear{};
                depthClear.depthStencil = {1.0f, 0};

                // C2 gate: skip the depth prepass and run a single forward
                // pass (color writes depth) when disabled in settings. Forced
                // on while solid RT paths run: the RT color variants have no
                // depth-write twin, and an unwritten depth target would break
                // the water/composite passes that sample it. The depth layout
                // transition below still runs in both modes (Instance 2 needs
                // it when Instance 1 is skipped).
                const bool skipSolidPrepass = !settings.solidDepthPrepass
                    && !settings.rtReflections && !settings.rtLocalShadows;
                if (settings.renderSolid && this->sceneRenderer->solidRenderer)
                    this->sceneRenderer->solidRenderer->setDeferredColorDepthWrite(skipSolidPrepass);

                // Transition solid depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the pre-pass.
                {
                    VkImage solidDepthImg = this->sceneRenderer->solidRenderer->getDepthImage(frameIdx);
                    if (solidDepthImg != VK_NULL_HANDLE) {
                        RendererUtils::transitionImageLayout(
                            solidCmd, solidDepthImg,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT);
                        setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
                    }
                }

                // ── Instance 1: Deferred depth pre-pass (no color attachment) ──
                // Skipped by the C2 gate (single forward pass instead).
                if (!skipSolidPrepass) {
                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView = this->sceneRenderer->solidRenderer->getDepthView(frameIdx);
                    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    depthAtt.clearValue = depthClear;

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {0, 0};
                    ri.renderArea.extent = {this->sceneRenderer->solidRenderer->getRenderWidth(), this->sceneRenderer->solidRenderer->getRenderHeight()};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 0;
                    ri.pColorAttachments = nullptr;
                    ri.pDepthAttachment = &depthAtt;

                    vkCmdBeginRendering(solidCmd, &ri);

                    {
                        VkViewport vp{0.0f, 0.0f, (float)this->sceneRenderer->solidRenderer->getRenderWidth(), (float)this->sceneRenderer->solidRenderer->getRenderHeight(), 0.0f, 1.0f};
                        vkCmdSetViewport(solidCmd, 0, 1, &vp);
                        VkRect2D sc{{0, 0}, {this->sceneRenderer->solidRenderer->getRenderWidth(), this->sceneRenderer->solidRenderer->getRenderHeight()}};
                        vkCmdSetScissor(solidCmd, 0, 1, &sc);
                    }

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 6);
                    if (settings.renderSolid) {
                        this->sceneRenderer->solidRenderer->drawDepth(solidCmd, this, getMainDescriptorSet());
                    }
                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 7);

                    vkCmdEndRendering(solidCmd);
                }

                // Transition solid color to COLOR_ATTACHMENT_OPTIMAL for the color pass.
                {
                    VkImage solidColorImg = this->sceneRenderer->solidRenderer->getColorImage(frameIdx);
                    if (solidColorImg != VK_NULL_HANDLE) {
                        RendererUtils::transitionImageLayout(
                            solidCmd, solidColorImg,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
                        setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 1);
                    }
                }

                // GPU droplet simulation: recorded into this same command
                // buffer before the color pass so the droplet draw below reads
                // the freshly integrated state (the sim ends with a
                // compute -> vertex/fragment barrier).
                dropletDrawCount = 0;
                if (settings.dropletsEnabled && particleSim && particleSim->ready()) {
                    if (settings.dropletResetSeconds > 0.0f) {
                        dropletSeconds += std::max(lastFrameDelta, 0.0f);
                        if (dropletSeconds >= settings.dropletResetSeconds) {
                            particleSim->requestReset();
                            dropletSeconds = 0.0;
                        }
                    }
                    if (physicsWidget && physicsWidget->resetRequested) {
                        physicsWidget->resetRequested = false;
                        particleSim->requestReset();
                        dropletSeconds = 0.0;
                    }
                    if (dropletShapeRadius != settings.dropletRadius) {
                        particleSim->setParticleShape(settings.dropletRadius, 1.0f);
                        dropletShapeRadius = settings.dropletRadius;
                    }
                    ParticlePhysicsSim::Config simCfg;
                    simCfg.particleCount = static_cast<uint32_t>(std::clamp(
                        settings.dropletCount, 1, static_cast<int>(ParticlePhysicsSim::kMaxParticles)));
                    simCfg.spawnRadius = settings.dropletSpawnRadius;
                    simCfg.spawnCenter = dropletSpawnCenter_;
                    simCfg.gravity = settings.dropletGravity;
                    simCfg.restitution = settings.dropletRestitution;
                    simCfg.friction = settings.dropletFriction;
                    simCfg.damping = settings.dropletDamping;
                    simCfg.waterLevel = 0.0f; // MainSceneLoader water box top
                    simCfg.particleRadius = settings.dropletRadius;
                    simCfg.solverIterations = settings.dropletSolverIterations;
                    simCfg.substeps = settings.dropletSubsteps;
                    simCfg.regionMin = dropletRegionMin;
                    simCfg.regionMax = dropletRegionMax;
                    // Fixed substep: the frame delta is split into `substeps`
                    // equal slices, clamped so a hitch can never tunnel a
                    // droplet through the SDF in one step.
                    const float frameDt = (dropletForcedDt > 0.0f)
                        ? dropletForcedDt
                        : std::clamp(lastFrameDelta, 0.0f, 0.1f);
                    const float substep = frameDt / static_cast<float>(std::max(1, simCfg.substeps));
                    particleSim->dispatch(solidCmd, substep, simCfg);
                    dropletDrawCount = simCfg.particleCount;
                }

                // ── Instance 2: Color pass (load depth from prepass) ──
                {
                    VkRenderingAttachmentInfo colorAtt{};
                    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtt.imageView = this->sceneRenderer->solidRenderer->getColorView(frameIdx);
                    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    colorAtt.clearValue = colorClear;

                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView = this->sceneRenderer->solidRenderer->getDepthView(frameIdx);
                    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    // C2 gate: CLEAR when the prepass was skipped (single
                    // forward pass, color writes depth), LOAD otherwise.
                    // STORE is kept in both modes: the water pass samples
                    // this depth later in the frame.
                    depthAtt.loadOp = skipSolidPrepass ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
                    // STORE (not DONT_CARE): the solid depth is sampled later in
                    // this same frame by the water pass (SSR/shore clamps,
                    // raster water-region depth) and by debug/preview views.
                    // DONT_CARE left the sampled image undefined after the color
                    // pass — on this driver it read as clear over the whole lake
                    // even though the depth prepass had written the bed, which
                    // cut the raster water depth off.
                    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    depthAtt.clearValue = depthClear;

                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {0, 0};
                    ri.renderArea.extent = {this->sceneRenderer->solidRenderer->getRenderWidth(), this->sceneRenderer->solidRenderer->getRenderHeight()};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &colorAtt;
                    ri.pDepthAttachment = &depthAtt;

                    vkCmdBeginRendering(solidCmd, &ri);

                    {
                        VkViewport vp{0.0f, 0.0f, (float)this->sceneRenderer->solidRenderer->getRenderWidth(), (float)this->sceneRenderer->solidRenderer->getRenderHeight(), 0.0f, 1.0f};
                        vkCmdSetViewport(solidCmd, 0, 1, &vp);
                        VkRect2D sc{{0, 0}, {this->sceneRenderer->solidRenderer->getRenderWidth(), this->sceneRenderer->solidRenderer->getRenderHeight()}};
                        vkCmdSetScissor(solidCmd, 0, 1, &sc);
                    }

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 8);
                    if (this->sceneRenderer->skyRenderer) {
                        SkySettings::Mode skyMode = this->sceneRenderer->getSkySettings().mode;
                        this->sceneRenderer->skyRenderer->render(this, solidCmd, getMainDescriptorSet(),
                            this->sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, viewProj, skyMode);
                    }
                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 9);

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 10);

                    if (settings.renderSolid) {
                        // Set=1 brush SDF params: PAINT/REMOVE evaluates the
                        // brush field at each fragment directly.
                        VkDescriptorSet brushParamsSet = this->sceneRenderer->brushRenderer->getParamsDescriptorSet(frameIdx);
                        this->sceneRenderer->solidRenderer->drawColor(solidCmd, this, getMainDescriptorSet(), brushParamsSet);
                        // C2: release the single-pass selector so later
                        // external draws never inherit the depth-write twin.
                        this->sceneRenderer->solidRenderer->setDeferredColorDepthWrite(false);

                        // Droplets: direct SDF ray marching in the same pass
                        // after the opaque geometry (depth test against the
                        // solid depth, no depth write, premultiplied coverage
                        // blend). Reads the state the dispatches above wrote.
                        if (settings.dropletsEnabled && dropletRenderer && particleSim &&
                            particleSim->ready() && dropletDrawCount > 0u) {
                            VkImageView skyView = this->sceneRenderer->skyRenderer
                                ? this->sceneRenderer->skyRenderer->getSkyView(frameIdx)
                                : VK_NULL_HANDLE;
                            dropletRenderer->draw(solidCmd, this, frameIdx, getMainDescriptorSet(),
                                                  particleSim->stateBuffer(), dropletDrawCount,
                                                  1.0f, settings.dropletIor, 0.0f, skyView);
                        }
                    }

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 11);

                    // Debug overlays on top
                    const bool showOctreeDebug = octreeExplorerWidget && octreeExplorerWidget->getShowDebugCubes();
                    if (showOctreeDebug) {
                        std::vector<DebugCubeRenderer::CubeWithColor> widgetCubes;
                        if (octreeExplorerWidget->isVisible()) {
                            const auto& wc = octreeExplorerWidget->getExpandedCubes();
                            widgetCubes.reserve(wc.size());
                            for (const auto& c : wc)
                                widgetCubes.push_back({BoundingBox(c.cube.getMin(), c.cube.getMax()), c.color});
                        }
                        this->sceneRenderer->debugCubeRenderer->renderOverlay(this, solidCmd, getMainDescriptorSet(), widgetCubes);
                    }

                    if (settings.renderSolid && settings.wireframeMode && this->sceneRenderer) {
                        this->sceneRenderer->solidRenderer->drawWireframeOverlay(solidCmd, this, getMainDescriptorSet());
                    }

                    vkCmdEndRendering(solidCmd);
                }

                // Transition solid color+depth to SHADER_READ_ONLY_OPTIMAL so the
                // composite (main CB) and the water pass can sample them after
                // tlSolid is signaled.
                {
                    VkImage solidColorImg = this->sceneRenderer->solidRenderer->getColorImage(frameIdx);
                    VkImage solidDepthImg = this->sceneRenderer->solidRenderer->getDepthImage(frameIdx);
                    uint32_t bc = 0;
                    VkImageMemoryBarrier2 barriers[2]{};

                    if (solidColorImg != VK_NULL_HANDLE) {
                        barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        barriers[bc].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                        barriers[bc].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                        barriers[bc].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        barriers[bc].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        barriers[bc].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        barriers[bc].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        barriers[bc].image = solidColorImg;
                        barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                        ++bc;
                    }
                    if (solidDepthImg != VK_NULL_HANDLE) {
                        barriers[bc].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                        barriers[bc].srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                        barriers[bc].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                        barriers[bc].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                        barriers[bc].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                        barriers[bc].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                        barriers[bc].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        barriers[bc].image = solidDepthImg;
                        barriers[bc].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
                        ++bc;
                    }
                    if (bc > 0) {
                        VkDependencyInfo dep{};
                        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        dep.imageMemoryBarrierCount = bc;
                        dep.pImageMemoryBarriers = barriers;
                        vkCmdPipelineBarrier2(solidCmd, &dep);
                    }
                    if (solidColorImg != VK_NULL_HANDLE)
                        setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
                    if (solidDepthImg != VK_NULL_HANDLE)
                        setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
                }

                // Submit to the dedicated solid queue. Wait on the cull
                // (visibleLods) and shadow (shadow map + restored UBO) results,
                // then signal tlSolid@v (registered so the main composite CB
                // waits on it). tlSolid@v is also waited by the water task.
                // Shadows disabled: nothing signals tlShadow, so the solid pass
                // waits tlCull@v directly instead (cull buffers + restored UBO).
                // The brush preview is an independent offscreen pass; its signal
                // (tlBrush) goes straight to the composite, not through the
                // solid pass. tlSolid is not registered for the composite
                // (implied by tlWater via Water).
                //
                // Solid SSR: SolidRenderer.frag samples the *previous* frame's solid
                // color/depth (bindings 19/20), so the solid CB additionally
                // waits on tlSolid@(v-1) — the previous frame's solid pass must
                // have completed before its images are marched.
                {
                    std::vector<VkSemaphore> solidWaitSemaphores = {settings.enableShadows ? tlShadow : tlCull};
                    std::vector<uint64_t> solidWaitValues = {v};
                    if (v > 0) {
                        solidWaitSemaphores.push_back(tlSolid);
                        solidWaitValues.push_back(v - 1);
                    }
                    app->submitCommandBufferAsyncToQueue(solidCmd, app->getSolidQueue(), &tlSolid, solidWaitSemaphores, false, {}, solidWaitValues, v, {}, false);
                }
                this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);
            });
            // Join the solid task before the main (composite) command buffer is recorded
            // and submitted, so semSolid is registered into m_extraWaitSemaphores and the
            // main CB waits on it (mirrors the cull / solid360 sync-join pattern).
            if (asyncSolidFuture.valid())
                asyncSolidFuture.get();
            // Join the sky task so semSky is registered before the solid360 task
            // (which samples the sky image) is submitted.
            if (asyncSkyFuture.valid())
                asyncSkyFuture.get();
        }
        // --- Solid360 360° capture: REMOVED (hybrid RT §12) ---
        // The old per-frame 6-face cubemap rasterization (6 culls + 6 raster
        // CBs + join + mip blits) no longer runs; Solid360Renderer and its
        // targets/sync/descriptors were deleted. Solid reflections and water
        // reflection/refraction are hardware ray tracing (proxy TLAS + water
        // RT pipeline, sky equirect on miss). Water waits tlSolid directly.

        // --- Vegetation pass on its own command buffer (signals semVeg) ---
        // Rendered to its own offscreen color+depth framebuffer (decoupled from
        // the solid pass). Runs after the shadow task on the worker, so the shadow
        // map it samples is current. Occlusion against solid geometry is resolved
        // at composite time (PostProcessRenderer.frag). Like the other async passes, it
        // uses a dedicated CommandBufferState and signals semVeg (registered so the
        // composite waits on it).
        // H7 + M12 (perf report 27): single veg-pass decision owned by the
        // renderer (VegetationRenderer::beginVegPass over wantDraw +
        // hasDrawContent). hasDrawContent() is main-thread readable, so the
        // outcome is known BEFORE either the veg or the water task is
        // enqueued. Skipped frames enqueue no veg task at all (no CB, no
        // tlVeg signal); both water submits wait tlCull@v instead of tlVeg@v
        // (see the water blocks). tlCull is signaled unconditionally every
        // frame, and skipped frames leave the veg targets stable
        // SHADER_READ_ONLY with the cleared bytes the warmup frames left —
        // exactly what water's reflection lookup expects from an empty
        // meadow. Transitions/warmup always run full; the tlVeg wait is
        // restored automatically on content transitions (vegRun true again).
        // The old main-thread duplicate (vegPrevWantDraw/vegWarmupRuns) is
        // removed; the renderer holds the only transition/warmup state.
        bool vegWillSignal = true; // function scope: read by the water submits below
        {
            const glm::vec3 vegCamPos = camera.getPosition();
            // vegWantDraw matches the worker's (drawBillboards ||
            // drawImpostors): grassImpostorsEnabled already folds in
            // impostorDistance > 0, and the worker below uses the captured
            // vegImpostorDist snapshot instead of re-reading settings (same
            // capture pattern as the other flags), so the skip condition and
            // the worker's empty branch coincide provably.
            const bool vegWantDraw = vegetationEnabled || grassImpostorsEnabled;
            const float vegImpostorDist = settings.impostorDistance;
            const bool vegHasContent = sceneRenderer->vegetationRenderer
                ? sceneRenderer->vegetationRenderer->hasDrawContent() : false;
            const bool vegRun = sceneRenderer->vegetationRenderer
                ? sceneRenderer->vegetationRenderer->beginVegPass(vegWantDraw, vegHasContent)
                : vegWantDraw;
            vegWillSignal = vegRun;
            if (vegRun) {
            asyncVegFuture = asyncThreadPool.enqueue([this, frameIdx, viewProj, vegetationEnabled, grassImpostorsEnabled, vegCamPos, vegImpostorDist, v]() {
                MyApp* app = this;
                VkCommandBuffer vegCmd = app->beginAsyncTask("vegetation");
                if (vegCmd == VK_NULL_HANDLE) return;
                // Reset the query slots owned by this command buffer (12-13) so
                // the vegetation GPU timestamps start clean. The solid CB owns
                // 6-11, water 14-15/20-21, SDF 22-23 and the main CB 16-19.
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                    vkCmdResetQueryPool(vegCmd, queryPools[frameIdx], 12, 2);
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);
                VkImageView vegColorView = sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getVegColorView(frameIdx) : VK_NULL_HANDLE;
                VkImageView vegDepthView = sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getVegDepthView(frameIdx) : VK_NULL_HANDLE;
                if (vegColorView == VK_NULL_HANDLE || vegDepthView == VK_NULL_HANDLE) {
                // Shadows disabled: no shadow task signals tlShadow, so wait the
                // cull timeline directly (same frame value).
                app->submitCommandBufferAsyncToQueue(vegCmd, app->getVegetationQueue(), &tlVeg,
                    {settings.enableShadows ? tlShadow : tlCull}, true, {}, {v}, v, {}, true);
                lastVegSignal = v; // valid consolidation wait target (flicker fix)
                    return;
                }
                // H7 + M12 (perf report 27): no worker-side skip here. The
                // main thread already ran the unified arbiter (vegRun, see
                // above); this task is only enqueued on run frames, so it
                // always records + submits + signals tlVeg@v below. Skipped
                // frames never reach this lambda, and water waits tlCull@v
                // instead (vegWillSignal).
                // Render area must match the vegetation offscreen targets' backing size
                // (which are resized to the swapchain extent in SceneRenderer::onSwapchainResized).
                // Using getWidth()/getHeight() here caused VUID-VkRenderingInfo-pNext-06079 when the
                // window size advanced past the (stale) vegetation offscreen extent.
                const uint32_t w = sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getVegWidth() : getSwapchainExtent().width;
                const uint32_t h = sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getVegHeight() : getSwapchainExtent().height;
                VkImage vegColorImg = sceneRenderer->vegetationRenderer->getVegColorImage(frameIdx);
                VkImage vegDepthImg = sceneRenderer->vegetationRenderer->getVegDepthImage(frameIdx);
                VkImageLayout vegColorOld = sceneRenderer->vegetationRenderer->getVegColorLayout(frameIdx);
                VkImageLayout vegDepthOld = sceneRenderer->vegetationRenderer->getVegDepthLayout(frameIdx);
                app->recordTransitionImageLayoutLayer(vegCmd, vegColorImg, app->getSwapchainImageFormat(), vegColorOld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 0, 1);
                app->recordTransitionImageLayoutLayer(vegCmd, vegDepthImg, VK_FORMAT_D32_SFLOAT, vegDepthOld, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
                sceneRenderer->vegetationRenderer->setVegColorLayout(frameIdx, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                sceneRenderer->vegetationRenderer->setVegDepthLayout(frameIdx, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

                auto beginVegInstance = [&](VkRenderingAttachmentInfo* colorAtt, VkRenderingAttachmentInfo* depthAtt, uint32_t colorCount) {
                    VkRenderingInfo ri{};
                    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset = {0, 0};
                    ri.renderArea.extent = {w, h};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = colorCount;
                    ri.pColorAttachments = (colorCount > 0) ? colorAtt : nullptr;
                    ri.pDepthAttachment = depthAtt;
                    vkCmdBeginRendering(vegCmd, &ri);
                    VkViewport vp{0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
                    vkCmdSetViewport(vegCmd, 0, 1, &vp);
                    VkRect2D sc{{0, 0}, {w, h}};
                    vkCmdSetScissor(vegCmd, 0, 1, &sc);
                };

                // Blade tufts and/or the captured impostors. The blades own
                // [0, impostorDistance) and the impostors take over beyond.
                const bool drawBillboards = vegetationEnabled;
                const bool drawImpostors = vegImpostorDist > 0.0f &&
                                           (vegetationEnabled || grassImpostorsEnabled);
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                    vkCmdWriteTimestamp(vegCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 12);
                if ((drawBillboards || drawImpostors) && sceneRenderer->vegetationRenderer) {
                    sceneRenderer->vegetationRenderer->recordReadBarriers(vegCmd);
                    if (drawBillboards) {
                        // Instance 1: depth prepass (no color attachment)
                        {
                            VkClearValue vegDepthClear{}; vegDepthClear.depthStencil = {1.0f, 0};
                            VkRenderingAttachmentInfo depthAtt{};
                            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                            depthAtt.imageView = vegDepthView;
                            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                            depthAtt.clearValue = vegDepthClear;
                            beginVegInstance(nullptr, &depthAtt, 0);
                            sceneRenderer->vegetationRenderer->drawDepth(this, vegCmd, vegCamPos);
                            vkCmdEndRendering(vegCmd);
                        }
                        // Instance 2: color + depth (depth loaded from prepass)
                        {
                            VkClearValue vegColorClear{}; vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                            VkRenderingAttachmentInfo colorAtt{};
                            colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                            colorAtt.imageView = vegColorView;
                            colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                            colorAtt.clearValue = vegColorClear;
                            VkRenderingAttachmentInfo depthAtt{};
                            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                            depthAtt.imageView = vegDepthView;
                            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                            beginVegInstance(&colorAtt, &depthAtt, 1);
                            sceneRenderer->vegetationRenderer->drawColor(this, vegCmd, vegCamPos,
                                                                         /*drawBillboards=*/true,
                                                                         /*drawImpostors=*/drawImpostors);
                            vkCmdEndRendering(vegCmd);
                        }
                    } else {
                        // Grass Raycast impostor-only: one color+depth CLEAR
                        // instance. No billboard depth prepass is needed — the
                        // impostor pipeline writes the captured depth itself
                        // (gl_FragDepth from the captured view depth).
                        VkClearValue vegColorClear{}; vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                        VkClearValue vegDepthClear{}; vegDepthClear.depthStencil = {1.0f, 0};
                        VkRenderingAttachmentInfo colorAtt{};
                        colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        colorAtt.imageView = vegColorView;
                        colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        colorAtt.clearValue = vegColorClear;
                        VkRenderingAttachmentInfo depthAtt{};
                        depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        depthAtt.imageView = vegDepthView;
                        depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        depthAtt.clearValue = vegDepthClear;
                        beginVegInstance(&colorAtt, &depthAtt, 1);
                        sceneRenderer->vegetationRenderer->drawColor(this, vegCmd, vegCamPos,
                                                                     /*drawBillboards=*/false,
                                                                     /*drawImpostors=*/true);
                        vkCmdEndRendering(vegCmd);
                    }
                } else {
                    // Clear the offscreen so the composite sees empty vegetation.
                    VkClearValue vegColorClear{}; vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                    VkClearValue vegDepthClear{}; vegDepthClear.depthStencil = {1.0f, 0};
                    VkRenderingAttachmentInfo colorAtt{};
                    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtt.imageView = vegColorView;
                    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    colorAtt.clearValue = vegColorClear;
                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView = vegDepthView;
                    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    depthAtt.clearValue = vegDepthClear;
                    beginVegInstance(&colorAtt, &depthAtt, 1);
                    vkCmdEndRendering(vegCmd);
                }

                // Transition back to SHADER_READ_ONLY for the composite.
                app->recordTransitionImageLayoutLayer(vegCmd, vegColorImg, app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                app->recordTransitionImageLayoutLayer(vegCmd, vegDepthImg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                sceneRenderer->vegetationRenderer->setVegColorLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                sceneRenderer->vegetationRenderer->setVegDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                    vkCmdWriteTimestamp(vegCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 13);

                // Run on the dedicated vegetation queue (separate from graphicsQueue) so it
                // parallelizes with the solid/shadow passes. Wait on the cull task's
                // GPU-written vegetation instance/indirect buffers (tlCull@v) so they are
                // visible before the veg pass reads them, and on the shadow task's timeline
                // signal (tlShadow@v) so the shadow map written by the shadow task (on
                // graphicsQueue) is visible before the veg pass samples it. Shadows
                // disabled: nothing signals tlShadow, so the wait becomes tlCull@v (the
                // shadow map is not sampled with shadowEffects.w == 0). The wait provides
                // the cross-queue (graphics→vegetation) memory dependency, and being the
                // same queue family no ownership transfer is required.
                app->submitCommandBufferAsyncToQueue(vegCmd, app->getVegetationQueue(), &tlVeg,
                    {settings.enableShadows ? tlShadow : tlCull}, true, {}, {v}, v, {}, true);
                // Publish the value the consolidation copy must wait on
                // (flicker fix). Worker-thread write; the frame thread reads
                // it only after joining this future.
                lastVegSignal = v;
            });
            } // if (vegRun): skipped frames enqueue nothing (H7, report 27)
        }

        // --- Mesh bounding boxes on its own command buffer (signals semBbox) ---
        // Renders to its own offscreen color+depth framebuffer (decoupled from the
        // solid pass) so it runs in parallel with the solid/vegetation shading on the
        // dedicated bboxQueue. Depends on the cull task's bbox compact/count buffers
        // (written by the terrain IR's IndirectRenderer.comp), so it waits on its own
        // cull-result semaphore (semCullBbox). When the overlay is disabled it only
        // clears the offscreen so the composite shows no bounding boxes.
        {
            const bool bboxEnabled = settings.showBoundingBoxes;
            asyncBboxFuture = asyncThreadPool.enqueue([this, frameIdx, bboxEnabled, v]() {
                // C3: same steady-state elision as the SDF task above.
                static bool bboxPrevEnabled = false;
                static uint32_t bboxRuns = 0;
                const bool bboxTransition = (bboxEnabled != bboxPrevEnabled);
                bboxPrevEnabled = bboxEnabled;
                if (!bboxEnabled && !bboxTransition && bboxRuns >= 3) return;
                ++bboxRuns;
                MyApp* app = this;
                VkCommandBuffer bboxCmd = app->beginAsyncTask("bbox");
                if (bboxCmd == VK_NULL_HANDLE) return;
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);
                if (this->sceneRenderer->boundingBoxRenderer) {
                    this->sceneRenderer->boundingBoxRenderer->renderToOffscreen(this, bboxCmd, app->getMainDescriptorSet(), frameIdx, bboxEnabled);
                }
                // Wait on semCullBbox (own binary semaphore, distinct from the main
                // CB's semMainCull) so the cull task's GPU-written bbox buffers are
                // visible before the bbox pass reads them; signal semBbox for the composite.
                app->submitCommandBufferAsyncToQueue(bboxCmd, app->getBoundingBoxQueue(), &tlBbox, {tlCull}, true, {}, {v}, v, {}, true);
            });
        }

        // Back-face depth + water geometry pass on a shared command buffer.
        // (waits semMainCull + semSolid360; signals semWater at the end of the task)
        // Frames whose previous cull produced zero visible water chunks take a
        // clear-only fast path instead (M8).
        if (waterEnabled && sceneRenderer) {
            asyncBackFaceFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, v, vegWillSignal]() {
                MyApp* app = this;
                VkCommandBuffer cmd = app->beginAsyncTask("backFace");
                if (cmd == VK_NULL_HANDLE) return;
                // Reset the query slots owned by the water pass (14-15) so the GPU
                // profiling timestamps below start from a clean state. The hybrid-RT
                // dispatch slots (20-21) are reset inside their own gate below, next
                // to the timestamps that write them.
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
                    vkCmdResetQueryPool(cmd, queryPools[frameIdx], 14, 2);
                }

                // ── Empty-water fast path (perf_report_19 M8) ──────────────────
                // waterPassEmpty is gated on the CPU-side resident water mesh
                // count, NOT on the GPU visible-count readback: that readback is
                // copied asynchronously and can lag several frames, so a stale
                // zero cleared the water targets while the current cull still
                // had visible chunks — already-loaded water flickered while
                // chunks streamed in. With zero resident meshes the pass could
                // not draw anything anyway, so the clear-only path is
                // unconditionally safe. Water-in-main writes the MAIN targets,
                // so it must never take this path.
                const size_t waterMeshCount = this->sceneRenderer->waterRenderer
                    ? this->sceneRenderer->waterRenderer->getIndirectRenderer().getMeshCount()
                    : 0;
                const bool waterPassEmpty = waterMeshCount == 0;
                if (waterPassEmpty && !settings.waterInMainPass) {
                    // Record ONLY the water-target clears (color/body/column to
                    // transparent black, depth to 1.0) plus their layout
                    // transitions. Skip the task-local cull, the back-face
                    // set/render, the water descriptor updates, the water
                    // geometry pass, the RT dispatch and timestamps 14/15.
                    this->sceneRenderer->waterRenderer->clearRenderTargets(this, cmd, frameIdx);
                    // Register tlWater for the composite directly (the empty
                    // fast path produces the water targets the composite
                    // samples). Submit with the same waits/signal as the full
                    // path so the composite chain stays valid.
                    // H7 (perf report 27): on veg-skipped frames the veg task
                    // enqueued nothing and signaled no tlVeg@v, so wait tlCull@v
                    // instead (signaled unconditionally every frame; the veg
                    // targets sit stable SHADER_READ_ONLY with cleared bytes).
                    // Same values, same DAG; one fewer submit on empty frames.
                    lastWaterSignal = v;
                    app->submitCommandBufferAsyncToQueue(cmd, app->getWaterQueue(), &tlWater,
                        {tlSolid, tlSky, vegWillSignal ? tlVeg : tlCull}, true, {}, {v, v, v}, v, {}, true);
                    return;
                }

                // Dedicated command-buffer state for this async command buffer (see cull task).
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);
                // Reuse a ring of pre-allocated per-task resources (descriptor
                // sets) instead of creating host-visible buffers every frame.
                // Slot safety: see the cachedBackfaceRing comment above -- the
                // previous submission using this slot (task N) has completed
                // before task N+ASYNC_RING_SIZE runs, so reusing the slot
                // cannot race with the GPU.
                IndirectRenderer &ind = this->sceneRenderer->waterRenderer->getIndirectRenderer();
                BackfaceSlot& slot = cachedBackfaceRing[ringBackface++ % ASYNC_RING_SIZE];

                VkDevice dev = app->getDevice();
                // ── Back-face demand gate (perf_report_19 C3) ─────────────────
                // The back-face depth pass exists only to measure the water
                // VOLUME (column thickness) for these per-layer consumers:
                //   enableWaves      -> TES shore-zone/gradient + shoaling depth
                //   enableFoam       -> thickness-zoned foam bands/contact line
                //   enableVolumetric -> scattering integrates over the column
                //   causticIntensity -> refracting-column inverse-Jacobian light
                //   enableRefraction -> Beer-Lambert thickness / tint absorption
                // Blur is NOT a consumer: the body/column it reads are computed
                // by the water fragment shader from the wave field regardless of
                // the back-face pass. When no layer needs the volume, skip the
                // cull, the water-depth descriptor set, the dummy patch and the
                // pass entirely; slot.pool/slot.waterDs stay untouched, and the
                // water set binds the 1x1 dummy depth (clear depth ->
                // hasValidBackFace=false, waterThickness=0).
                bool waterVolumeNeeded = false;
                for (const WaterSettings& wp : waterParams) {
                    waterVolumeNeeded = waterVolumeNeeded || wp.enableWaves || wp.enableFoam
                        || wp.enableVolumetric || wp.causticIntensity > 0.001f || wp.enableRefraction;
                }
                // M11 (perf report 21): the back-face pass always reuses the
                // main water cull output now. The shadow cascade cull used to
                // zero the main compact/count buffers (forcing a task-local
                // re-cull with shadows on), but prepareCull skips those fills
                // when doMain==false — the cascade dispatch never writes the
                // main streams (pc.doMain==0) — so the cull-task results
                // survive for this task under the same tlSolid-transitive
                // ordering proof as the old shadows-off path. One full
                // capacity dispatch saved per frame.

                // Water-depth descriptor set for THIS task: pre-allocated per ring slot
                // and rewritten each frame before submission. Reuse is safe because the
                // slot's previous submission, which bound this set on the GPU, has
                // completed before we rewrite it (see the cachedBackfaceRing comment).
                // A dedicated set per slot keeps the async back-face pass from sharing
                // the per-frame set with the main command buffer (which would require
                // UPDATE_AFTER_BIND and trip GPU-assisted validation's descriptor-count
                // check).
                VkDescriptorSet asyncWaterDs = VK_NULL_HANDLE;
                // Demand gate: no volume consumer means nothing reads the depth,
                // so skip the water-depth set creation/update as well. It stays
                // lazily created on the first frame the volume is needed again.
                if (waterVolumeNeeded) {
                    VkImageView bfBack = (this->sceneRenderer->backFaceRenderer) ? this->sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
                    // Hybrid RT: back-face set carries RT outputs + sky (no cubemap).
                    // isSupported() = device capability, runtimeEnabled() = runtime
                    // policy: while every ray path is off the non-RT fragment variant
                    // compiles these bindings out, so leave them null and let the
                    // renderer bind its 1x1 dummies (no real RT output referenced).
                    VkImageView bfRefl = VK_NULL_HANDLE, bfRefr = VK_NULL_HANDLE, bfSky = VK_NULL_HANDLE;
                    if (this->sceneRenderer->rayTracing
                        && this->sceneRenderer->rayTracing->isSupported()
                        && this->sceneRenderer->rayTracing->runtimeEnabled()) {
                        bfRefl = this->sceneRenderer->rayTracing->getReflectionView();
                        bfRefr = this->sceneRenderer->rayTracing->getRefractionView();
                    }
                    if (this->sceneRenderer->skyRenderer)
                        bfSky = this->sceneRenderer->skyRenderer->getSkyView(frameIdx);
                    VkDescriptorSetLayout wdsLayout = this->sceneRenderer->waterRenderer->getWaterDepthDescriptorSetLayout();
                    if (wdsLayout != VK_NULL_HANDLE) {
                        if (slot.pool == VK_NULL_HANDLE) {
                            // Per-slot pool (maxSets=1) so the set is allocated once and
                            // rewritten every task; flags mirror the renderer's async
                            // pool (the water-depth layout has no UPDATE_AFTER_BIND
                            // bindings, so none is required here).
                            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 };
                            VkDescriptorPoolCreateInfo pci{};
                            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                            pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
                            pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                            if (vkCreateDescriptorPool(dev, &pci, nullptr, &slot.pool) == VK_SUCCESS) {
                                app->resources.addDescriptorPool(slot.pool, "cachedBackfaceWaterDepth pool");
                                VkDescriptorSetAllocateInfo ai{};
                                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                                ai.descriptorPool = slot.pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &wdsLayout;
                                if (vkAllocateDescriptorSets(dev, &ai, &slot.waterDs) != VK_SUCCESS) {
                                    slot.waterDs = VK_NULL_HANDLE; // retried on a later task
                                } else {
                                    app->resources.addDescriptorSet(slot.waterDs, "cachedBackfaceWaterDepth DS");
                                }
                            }
                        }
                        if (slot.waterDs != VK_NULL_HANDLE) {
                            updateWaterSceneTextures(slot.waterDs, bfBack, bfRefl, bfRefr, bfSky, frameIdx);
                            asyncWaterDs = slot.waterDs;
                        }
                    }
                }

                // Patch binding #0 (back-face depth) to the dummy depth to avoid
                // the back-face pass reading-from the same image it writes-to as
                // depth attachment (SYNC-HAZARD READ_AFTER_WRITE).
                if (asyncWaterDs != VK_NULL_HANDLE) {
                    WaterBackFaceRenderer* bfr = this->sceneRenderer->backFaceRenderer.get();
                    if (bfr && bfr->getDummyDepthView() != VK_NULL_HANDLE) {
                        bfr->patchBinding0(asyncWaterDs, bfr->getDummyDepthView());
                    }
                }

                // Render back-face pass using the cull results. Ordering proof
                // for the shared cull reuse: this command buffer waits
                // tlSolid@v (submit at the end of the task), tlSolid is signaled
                // by the solid pass which waits tlShadow, and tlShadow is
                // signaled by the shadow task after waiting tlCull (with shadows
                // disabled the solid pass waits tlCull@v directly, so tlCull is
                // still transitively implied here). The main
                // water cull ran in the cull task BEFORE tlCull was signaled,
                // and a timeline semaphore wait makes those SSBO writes visible
                // to this queue, so the main cull output is complete before this
                // draw executes. The shadow cascade cull preserves these
                // buffers (prepareCull skips the main-output fills when
                // doMain==false), so the proof holds with shadows on. The
                // later water geometry pass reads the same
                // compact/visible buffers in this same command buffer -- a
                // read-read with no hazard.
                auto tBackface = std::chrono::high_resolution_clock::now();
                if (waterVolumeNeeded && this->sceneRenderer->backFaceRenderer) {
                    // M11: always the main water cull output (see above).
                    VkBuffer bfCompact = ind.getCurrentCompactBuffer();
                    VkBuffer bfVisible = ind.getCurrentVisibleCountBuffer();
                    this->sceneRenderer->backFaceRenderer->render(app, cmd, frameIdx,
                                                ind,
                                                this->sceneRenderer->waterRenderer->getWaterGeometryPipelineLayout(),
                                                app->getMainDescriptorSet(),
                                                asyncWaterDs,
                                                bfCompact,
                                                bfVisible);
                }

                this->profileBackface = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - tBackface).count();

                // ── Water geometry pass (same command buffer as the back-face pass) ──
                // The back-face depth (binding 0 of the water set) was produced earlier
                // in THIS command buffer and is already in SHADER_READ_ONLY_OPTIMAL (the
                // back-face pass transitions it after writing). The RT reflection /
                // refraction outputs + sky ride along in the same set (bindings 1-3).
                if (this->sceneRenderer->waterRenderer) {
                    auto& waterIR = this->sceneRenderer->waterRenderer->getIndirectRenderer();
                    waterIR.acquireBuffers(cmd);
                    // Demand gate: when no layer consumes the volume the back-face
                    // pass did not run this frame, so its depth image holds stale
                    // data. Bind the 1x1 dummy depth instead: it reads as clear
                    // depth -> hasValidBackFace=false, waterThickness=0, matching
                    // "no volume" (the back-face set binding 0 is written with the
                    // same dummy when the pass runs).
                    VkImageView wBack = VK_NULL_HANDLE;
                    if (this->sceneRenderer->backFaceRenderer) {
                        wBack = waterVolumeNeeded
                            ? this->sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIdx)
                            : this->sceneRenderer->backFaceRenderer->getDummyDepthView();
                    }
                    // Same capability-vs-policy gate as the back-face set: with the
                    // RT runtime off the water fragment variant never samples these
                    // bindings, so bind the renderer's dummies instead of the real
                    // RT output views (H5).
                    VkImageView wRefl = VK_NULL_HANDLE, wRefr = VK_NULL_HANDLE;
                    if (this->sceneRenderer->rayTracing
                        && this->sceneRenderer->rayTracing->isSupported()
                        && this->sceneRenderer->rayTracing->runtimeEnabled()) {
                        wRefl = this->sceneRenderer->rayTracing->getReflectionView();
                        wRefr = this->sceneRenderer->rayTracing->getRefractionView();
                    }
                    // Dedicated per-slot set for the water geometry pass so the REAL
                    // back-face depth can be bound at binding 0 (the back-face pass used
                    // a different set with binding 0 patched to the dummy depth).
                    if (slot.waterDs2 == VK_NULL_HANDLE && slot.poolW == VK_NULL_HANDLE) {
                        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 };
                        VkDescriptorPoolCreateInfo pci{};
                        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                        pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
                        pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                        if (vkCreateDescriptorPool(dev, &pci, nullptr, &slot.poolW) == VK_SUCCESS) {
                            app->resources.addDescriptorPool(slot.poolW, "cachedBackfaceWaterGeom pool");
                            VkDescriptorSetLayout wdsLayout = this->sceneRenderer->waterRenderer->getWaterDepthDescriptorSetLayout();
                            if (wdsLayout != VK_NULL_HANDLE) {
                                VkDescriptorSetAllocateInfo ai{};
                                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                                ai.descriptorPool = slot.poolW; ai.descriptorSetCount = 1; ai.pSetLayouts = &wdsLayout;
                                if (vkAllocateDescriptorSets(dev, &ai, &slot.waterDs2) != VK_SUCCESS)
                                    slot.waterDs2 = VK_NULL_HANDLE;
                                else app->resources.addDescriptorSet(slot.waterDs2, "cachedBackfaceWaterGeom DS");
                            }
                        }
                    }
                    // Water RT shading gate: global ray-path toggles AND the
                    // per-material water layer flags. Drives BOTH the water
                    // pipeline variant (heavy ray-query fragment shader vs. the
                    // cheaper non-RT shader) and the async RT dispatch below.
                    bool anyLayerRefl = false;
                    bool anyLayerRefr = false;
                    for (const WaterSettings& wp : waterParams) {
                        anyLayerRefl = anyLayerRefl || wp.enableReflection;
                        anyLayerRefr = anyLayerRefr || wp.enableRefraction;
                    }
                    const bool waterRtNeeded =
                        (settings.rtWaterReflections && anyLayerRefl) ||
                        (settings.rtRefractions && anyLayerRefr) ||
                        settings.rtWaterDepth;
                    // The async RT pipeline now produces refraction/thickness
                    // only (its reflection output is retired), so dispatch it
                    // only when a refraction path can consume it.
                    const bool waterPipeNeeded =
                        settings.rtWaterPipeline && settings.rtRefractions && anyLayerRefr;
                    if (this->sceneRenderer->waterRenderer) {
                        this->sceneRenderer->waterRenderer->setRtShadingEnabled(waterRtNeeded);
                        // Global tessellation gate (perf_report_19 C1): when the
                        // preset disables tessellation the water passes must use
                        // the non-tessellated pipeline; delivered per frame so a
                        // preset switch takes effect immediately.
                        this->sceneRenderer->waterRenderer->setTessellationEnabled(settings.tessellationEnabled);
                        if (this->sceneRenderer->backFaceRenderer) {
                            this->sceneRenderer->backFaceRenderer->setTessellationEnabled(settings.tessellationEnabled);
                        }
                        // Global feature gates delivered via the water render
                        // UBO: they apply in both fragment variants, so
                        // "refraction off" really disables the sky fallback
                        // too, and blur requires the global toggle plus the
                        // per-material flag.
                        this->sceneRenderer->waterRenderer->setRtFeatureFlags(
                            settings.rtWaterReflections, settings.rtRefractions,
                            settings.blurEnabled);
                    }
                    // Global tessellation gate for the solid family (perf
                    // report 21 C1): mirrors the water gate above. Delivered
                    // per frame so a preset switch takes effect immediately;
                    // the selectors fall back to the tessellated pipelines
                    // wherever no no-tess twin exists (RT/profiling/brush),
                    // which is always correct.
                    if (this->sceneRenderer->solidRenderer) {
                        this->sceneRenderer->solidRenderer->setTessellationEnabled(settings.tessellationEnabled);
                    }

                    if (slot.waterDs2 != VK_NULL_HANDLE) {
                        VkImageView wsky = (this->sceneRenderer->skyRenderer)
                            ? this->sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
                        // SSR source: the solid pass color/depth for this frame.
                        // The water task waits on tlSolid, and the solid pass
                        // ends both images in SHADER_READ_ONLY_OPTIMAL.
                        updateWaterSceneTextures(slot.waterDs2, wBack, wRefl, wRefr, wsky, frameIdx);
                        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 14);
                        if (settings.waterInMainPass && this->sceneRenderer->solidRenderer
                            && this->sceneRenderer->waterRenderer->getWaterMainPipeline() != VK_NULL_HANDLE) {
                            // Phase-1 water-in-main: blend water directly into the
                            // main solid color/depth targets (LOAD-preserved) with
                            // the alpha-blended water pipeline instead of the
                            // separate water target pair. The current main targets
                            // are the attachments here, so the in-trace screen
                            // lookups bind the PREVIOUS frame's solid/veg views
                            // (1-frame latency, same convention as the RT pipeline
                            // outputs); back-face depth stays the current frame's
                            // (written earlier on this command buffer) and sky is
                            // current. The composite skips its water branch via
                            // the zeroed dummy water view.
                            constexpr uint32_t kFif = VulkanApp::MAX_FRAMES_IN_FLIGHT;
                            const uint32_t prevIdx = (frameIdx + kFif - 1u) % kFif;
                            updateWaterSceneTextures(slot.waterDs2, wBack, wRefl, wRefr, wsky, prevIdx);
                            // The bound solid depth is the PREVIOUS frame's, so
                            // the shader-side occlusion rejection (C5) must stay
                            // off here; the main targets' hardware depth test
                            // already rejects occluded water.
                            this->sceneRenderer->waterRenderer->setSolidDepthCurrentFrame(false);
                            this->sceneRenderer->waterRenderer->renderMainTargets(this, cmd, frameIdx,
                                this->sceneRenderer->solidRenderer->getColorImage(frameIdx),
                                this->sceneRenderer->solidRenderer->getColorView(frameIdx),
                                this->sceneRenderer->solidRenderer->getDepthImage(frameIdx),
                                this->sceneRenderer->solidRenderer->getDepthView(frameIdx),
                                slot.waterDs2);
                        } else {
                            // Offscreen path: the water task waits tlSolid, so the
                            // bound solid depth is this frame's and the shader may
                            // reject terrain-occluded fragments (C5).
                            this->sceneRenderer->waterRenderer->setSolidDepthCurrentFrame(true);
                            this->sceneRenderer->waterRenderer->renderPass(this, cmd, frameIdx,
                                settings.waterWireframeMode, this->mainTime, wsky, slot.waterDs2);
                        }
                        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 15);
                        // Transition water geometry depth to SRO for the compositor.
                        // (Water-in-main never writes it: skipped there.)
                        VkImage wgdImg = settings.waterInMainPass ? VK_NULL_HANDLE
                                                                  : this->sceneRenderer->waterRenderer->getWaterGeomDepthImage(frameIdx);
                        if (wgdImg != VK_NULL_HANDLE) {
                            app->recordTransitionImageLayoutLayer(cmd, wgdImg, VK_FORMAT_D32_SFLOAT,
                                this->sceneRenderer->waterRenderer->getWaterGeomDepthLayout(frameIdx),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
                            this->sceneRenderer->waterRenderer->setWaterGeomDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                        }
                        // Hybrid RT water dispatch (same CB, AFTER the geometry
                        // depth reached SHADER_READ_ONLY above — the rgen samples
                        // it via the RT set, so dispatching earlier would sample
                        // DEPTH_STENCIL_ATTACHMENT (VUID layout mismatch)).
                        // Outputs feed NEXT frame's water shading (1-frame
                        // latency, same-queue ordered). Own timestamps (20-21).
                        // waterPipeNeeded covers the global + per-layer
                        // refraction gates; skip the dispatch when no pipe
                        // consumer is enabled (reflection no longer uses it).
                        if (waterPipeNeeded &&
                            this->sceneRenderer->rayTracing &&
                            this->sceneRenderer->rayTracing->isPipelineReady() &&
                            settings.waterEnabled) {
                            // Reset the dispatch's own slots here, next to the
                            // timestamps that write them, so a skipped dispatch
                            // records no reset at all.
                            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
                                vkCmdResetQueryPool(cmd, queryPools[frameIdx], 20, 2);
                                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 20);
                            }
                            this->sceneRenderer->rayTracing->dispatchWaterRT(this, cmd, frameIdx,
                                uboStatic.invViewProjection, uboStatic.viewPosition);
                            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 21);
                        }
                    }
                }

                // Submit to the dedicated water queue. The ring slot's buffers/set are
                // NOT defer-destroyed: they are reused ASYNC_RING_SIZE tasks later, by
                // which time this submission has completed (guaranteed by the frame-fence
                // chain described on cachedBackfaceRing). Signal tlWater@v.
                // Water waits on the solid pass (tlSolid) and the sky (tlSky,
                // sampled by water shading + the RT dispatch); tlCull / tlShadow
                // are transitively implied — dropped. tlWater is registered as a
                // persistent composite timeline wait so the water targets are
                // synchronized with the composite. Water-in-main never samples
                // the water targets in the composite and is not registered.
                // Wait on the vegetation pass too: WaterRenderer.frag's reflection
                // lookup samples the vegetation color/depth targets, which are
                // written on the vegetation queue and signaled by tlVeg@v.
                // Record the value so a later RT AS build waits for this
                // frame's water consumers before overwriting TLAS/metas.
                lastWaterSignal = v;
                const bool registerWaterForComposite = !settings.waterInMainPass;
                // H7 (perf report 27): tlCull@v substitutes for tlVeg@v on
                // veg-skipped frames (no veg submit/signal those frames).
                app->submitCommandBufferAsyncToQueue(cmd, app->getWaterQueue(), &tlWater, {tlSolid, tlSky, vegWillSignal ? tlVeg : tlCull},
                    registerWaterForComposite, {}, {v, v, v}, v, {}, registerWaterForComposite);

            });
        }

        // Wait for the async back-face+water task to finish recording/submitting so
        // its local semaphores stay valid and no two threads bind the same descriptor
        // set concurrently. get() rethrows task exceptions (wait() would swallow them,
        // silently leaving semWater unsignaled and dropping the water pass for the frame).
        if (asyncBackFaceFuture.valid()) {
            try {
                asyncBackFaceFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] back-face task failed, skipping back-face pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] back-face task failed with unknown error, skipping back-face pass this frame" << std::endl;
            }
        }

        // --- SDF debug cubes + generic SDF volumes (fire/grass) on one CB
        // (signals tlSdf) ---
        // Enqueued AFTER the back-face/water join: the submit waits on
        // tlWater, and a timeline wait is only valid once the water signal
        // submit is guaranteed to precede it (same main-thread join pattern
        // as asyncShadowFuture -> solid). Both passes render to their own
        // offscreens on the dedicated sdfQueue. Debug cubes depend on the
        // cull task's SDF buffers (tlCull); the generic SdfRenderer is
        // self-contained (own grid upload) and shares the same submit so only
        // one tlSdf signal is needed.
        {
            const bool sdfEnabled = settings.showSDFDebug;
            // M11 (report 25): content-gated SDF submit, mirroring the
            // report-22 SDF-task/bbox elision. An idle SDF frame (debug cubes
            // off, no generic containers, ray-march gate off) records only
            // clears, so past warmup and toggle transitions the task is not
            // even enqueued: no draw, no depth-preload copy, no
            // submit/signal. Skipped frames register no tlSdf value and the
            // composite only waits on registered timelines, so the DAG is
            // unchanged. Target layouts stay stable: the last recorded frame
            // left them cleared + SHADER_READ_ONLY.
            // NOTE: getContainerCount() is read lock-free (one frame stale at
            // most); a stale zero only delays newly-appeared content by one
            // frame, never a validation error.
            const bool sdfHasContent = this->sceneRenderer && this->sceneRenderer->sdfRenderer &&
                this->sceneRenderer->sdfRenderer->getContainerCount() > 0;
            const bool sdfIdle = !sdfEnabled && !sdfHasContent && !settings.rayMarchingEnabled;
            static bool sdfPrevIdle = false;
            static uint32_t sdfWarmupRuns = 0;
            const bool sdfTransition = (sdfIdle != sdfPrevIdle);
            sdfPrevIdle = sdfIdle;
            const bool sdfSkipSubmit = sdfIdle && !sdfTransition && sdfWarmupRuns >= 3;
            if (!sdfSkipSubmit) {
            if (sdfWarmupRuns < 3) ++sdfWarmupRuns;
            asyncSdfFuture = asyncThreadPool.enqueue([this, frameIdx, sdfEnabled, v]() {
                // SDF fire is always on, so the task runs every frame it is
                // enqueued (no steady-state elision inside the task: the same
                // CB carries the fire render when content exists).
                // The debug-cubes pass is gated by sdfEnabled internally.
                MyApp* app = this;
                auto tSdf = std::chrono::high_resolution_clock::now();
                VkCommandBuffer sdfCmd = app->beginAsyncTask("sdf");
                if (sdfCmd == VK_NULL_HANDLE) return;
                // Reset the query slots owned by this command buffer (22-23) so
                // the SDF engine GPU timestamps start clean. The solid CB owns
                // 6-11, vegetation 12-13, water 14-15/20-21 and the main CB 16-19.
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                    vkCmdResetQueryPool(sdfCmd, queryPools[frameIdx], 22, 2);
                CommandBufferState taskState;
                this->sceneRenderer->setCmdState(&taskState);
                if (this->sceneRenderer->debugSDFRenderer) {
                    this->sceneRenderer->debugSDFRenderer->render(this, sdfCmd, app->getMainDescriptorSet(), frameIdx, sdfEnabled);
                }
                // Generic SDF volume (fire/grass): GPU-procedural animation
                // needs no CPU scene edits; just stream the time uniform per
                // frame. The scene slot + depth bindings were already
                // prepared on the main thread (before the shadow pass, which
                // also binds this set); prepareCull here only records the
                // HOST->shader barriers for this CB.
                if (this->sceneRenderer->sdfRenderer) {
                    float t = this->mainTime * (this->raymarchWidget ? this->raymarchWidget->timeScale : 1.0f);
                    this->sceneRenderer->sdfRenderer->setFrame(frameIdx);
                    this->sceneRenderer->sdfRenderer->updateParams(t, frameIdx);
                    // SDF engine GPU interval (slots 22-23): the host->shader
                    // barriers + the ray-march pass.
                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(sdfCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 22);
                    this->sceneRenderer->sdfRenderer->prepareCull(sdfCmd);
                    this->sceneRenderer->sdfRenderer->render(this, sdfCmd, app->getMainDescriptorSet(), frameIdx, true);
                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(sdfCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 23);
                }
                // Wait on tlCull (cull buffers), tlSolid (solid depth) and
                // tlWater (water surface depth, both sampled for occlusion
                // after their SHADER_READ_ONLY transitions); signal tlSdf for
                // the composite.
                app->submitCommandBufferAsyncToQueue(sdfCmd, app->getSdfQueue(), &tlSdf,
                    {tlCull, tlSolid, tlWater}, true, {}, {v, v, v}, v, {}, true);
                this->profileSdfCpu = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - tSdf).count();
            });
            } // !sdfSkipSubmit (M11 elision: no CB, no tlSdf signal)
        }
        // Wait for the shadow + vegetation async tasks (same rationale as above:
        // keeps their signal semaphores valid and avoids concurrent descriptor-set
        // binds on the shared renderers).
        if (asyncShadowFuture.valid()) {
            try {
                asyncShadowFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] shadow task failed, skipping shadow pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] shadow task failed with unknown error, skipping shadow pass this frame" << std::endl;
            }
        }
        if (asyncVegFuture.valid()) {
            try {
                asyncVegFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] vegetation task failed, skipping vegetation pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] vegetation task failed with unknown error, skipping vegetation pass this frame" << std::endl;
            }
        }
        if (asyncSdfFuture.valid()) {
            try {
                asyncSdfFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] sdf task failed, skipping SDF pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] sdf task failed with unknown error, skipping SDF pass this frame" << std::endl;
            }
        }
        if (asyncBboxFuture.valid()) {
            try {
                asyncBboxFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] bbox task failed, skipping bounding-box pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] bbox task failed with unknown error, skipping bounding-box pass this frame" << std::endl;
            }
        }
        // Restore the per-frame state for the main CB's continued recording.
        this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);

        // ── Water geometry pass ──
        // Now recorded on its OWN async command buffer (combined with the back-face
        // pass in the "backface/water" task) so it runs in parallel with the solid
        // and vegetation shading above. It waits on semMainCull (current-frame
        // visibleLods from the cull task), semSolid360 (solid360 cubemap), and the
        // back-face depth it consumes is produced earlier in the SAME command buffer,
        // avoiding any cross-CB layout hazard. It signals semWater, which
        // m_extraWaitSemaphores makes the composite (below) wait on.

        // ── Video layer pass (Solid -> Water -> Video) ──
        // Recorded on the main command buffer: the submit already waits on all
        // async passes and on the video-array uploads (extra wait semaphores),
        // so the composite sees current frames and no upload writes the array
        // while this pass samples it.
        if (sceneRenderer && sceneRenderer->videoRenderer) {
            sceneRenderer->videoRenderer->draw(commandBuffer, this, frameIdx,
                                               getMainDescriptorSet(),
                                               camera.getViewProjectionMatrix(),
                                               camera.getPosition(),
                                               settings.lodBias,
                                               settings.maxTargetLod);
        }

        profileCpuRecord = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuRecordT0).count();
    }

    blender_export::Extras collectBlenderExportExtras() {
        blender_export::Extras extras;
        if (!sceneRenderer || !sceneRenderer->sdfRenderer) return extras;
        const sdf_gpu::SdfScene& scene = sceneRenderer->sdfRenderer->exportScene();
        extras.sdfDefinitions.reserve(scene.definitions().size());
        for (const SdfDefinition& d : scene.definitions()) {
            extras.sdfDefinitions.push_back(
                {static_cast<uint32_t>(d.prim), d.params0, d.params1});
        }
        extras.sdfMaterials.reserve(scene.materials().size());
        for (const SdfMaterial& m : scene.materials()) {
            extras.sdfMaterials.push_back({m.baseColor, m.emission, m.emissionIntensity,
                                           m.roughness, m.metallic, m.opacity,
                                           m.textureLayer, m.textureTiling});
        }
        extras.sdfInstances.reserve(scene.instances().size());
        for (const SdfInstance& i : scene.instances()) {
            extras.sdfInstances.push_back({i.position, i.rotation, i.scale, i.heightScale,
                                           i.radiusScale, i.intensity, i.seed,
                                           i.defIdx, i.matIdx});
        }
        return extras;
    }

    void startBlenderExport(const std::string& path) {
        if (blenderExportRunning_.exchange(true)) {
            std::cout << "[MyApp] Blender export already running\n";
            return;
        }
        if (!world || useRemote) {
            // Same guard as Save Scene: exporting the empty local scene while
            // rendering a remote one would write the wrong data.
            std::cerr << "[MyApp] Blender export ignored in remote mode\n";
            blenderExportRunning_.store(false);
            return;
        }
        blender_export::Options options;
        options.outputPath = path;
        options.cancel = &blenderExportCancel_;
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            options.billboardCount = sceneRenderer->vegetationRenderer->billboardCount;
            options.billboardScale = sceneRenderer->vegetationRenderer->billboardScale;
        }
        options.lightDirection = light.getDirection();
        // Carry the live engine view into the Blender scene: the generated
        // camera starts exactly where the app camera is looking.
        options.haveCamera = true;
        options.cameraPosition = camera.getPosition();
        options.cameraTarget = camera.getPosition() + camera.getForward() * 2048.0f;
        options.cameraFovYDegrees = camera.getFovYDegrees();
        const blender_export::Extras extras = collectBlenderExportExtras();

        blenderExportCancel_.store(false);
        blenderExportFinished_.store(false);
        blenderExportProgress_.store(0.0f);
        {
            std::lock_guard<std::mutex> lock(blenderExportMutex_);
            blenderExportPhase_ = "starting";
            blenderExportStats_ = blender_export::Stats{};
        }
        std::cout << "[MyApp] Blender export -> " << path << " (background)\n";
        blenderExportThread_ = std::thread([this, options, extras]() {
            const blender_export::Stats stats = blender_export::exportSceneToBlender(
                world->scene(), options, extras,
                [this](float fraction, const std::string& phase) {
                    blenderExportProgress_.store(fraction);
                    std::lock_guard<std::mutex> lock(blenderExportMutex_);
                    blenderExportPhase_ = phase;
                });
            {
                std::lock_guard<std::mutex> lock(blenderExportMutex_);
                blenderExportStats_ = stats;
            }
            blenderExportRunning_.store(false);
            blenderExportFinished_.store(true);
        });
    }

    void pollBlenderExport() {
        if (!blenderExportFinished_.exchange(false)) return;
        if (blenderExportThread_.joinable()) blenderExportThread_.join();
        blenderExportStatusPending_ = true;
        std::lock_guard<std::mutex> lock(blenderExportMutex_);
        if (blenderExportStats_.ok) {
            const std::string& wrote = blenderExportStats_.blendWritten
                                            ? blenderExportStats_.blendPath
                                            : blenderExportStats_.outputPath;
            std::cout << "[MyApp] Blender export ok: " << wrote << std::endl;
            if (!blenderExportStats_.warning.empty()) {
                std::cerr << "[MyApp] warning: " << blenderExportStats_.warning << std::endl;
            }
        } else {
            std::cerr << "[MyApp] Blender export failed: " << blenderExportStats_.error
                      << std::endl;
        }
    }

    void renderImGui() override {
        static char sceneFolderBuf[512] = "scenes/default.scene";

        // VULKAN_HIDE_UI=1: skip the widget/menu UI entirely (headless captures
        // for diagnostics: the ImGui windows otherwise cover the 3D view).
        static const bool hideUi = std::getenv("VULKAN_HIDE_UI") != nullptr;
        if (hideUi) return;

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Generate Map")) generateMapPending = true;
                if (ImGui::MenuItem("Action")) action();
                ImGui::Separator();
                if (ImGui::MenuItem("Save Scene...")) scenePicker_.open(sceneFolderBuf, true,  "Mode: Save");
                if (ImGui::MenuItem("Load Scene...")) scenePicker_.open(sceneFolderBuf, false, "Mode: Load");
                ImGui::Separator();
                if (ImGui::MenuItem("Export to Blender (.blend)..."))
                    blenderPicker_.open(blenderExportBuf_, true, "Mode: Export");
                ImGui::Separator();
                if (ImGui::MenuItem("Exit")) requestClose();
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Demo", NULL, &imguiShowDemo);
                ImGui::MenuItem("Show Profiling", NULL, &profilingEnabled);
                if (ImGui::MenuItem("Fullscreen", "F11", isFullscreen)) {
                    toggleFullscreen();
                }
                ImGui::EndMenu();
            }
            // Widget menu
            widgetManager.renderMenu();
            ImGui::EndMainMenuBar();

            std::filesystem::path chosenScene;
            if (scenePicker_.render(chosenScene)) {
                std::string s = chosenScene.string();
                if (s.size() < sizeof(sceneFolderBuf)) {
                    std::memcpy(sceneFolderBuf, s.c_str(), s.size());
                    sceneFolderBuf[s.size()] = '\0';
                }
                if (scenePicker_.isSaveMode()) {
                    // Saving the (empty) local scene over a file while
                    // rendering remotely would destroy data: refuse.
                    if (world && !useRemote) world->scene().save(sceneFolderBuf, &settings);
                    else std::cout << "[MyApp] save ignored in remote mode\n";
                } else {
                    pendingLoadPath = sceneFolderBuf;
                    loadScenePending = true;
                }
            }

            std::filesystem::path chosenGlb;
            if (blenderPicker_.render(chosenGlb)) {
                const std::string s = chosenGlb.string();
                if (s.size() < sizeof(blenderExportBuf_)) {
                    std::memcpy(blenderExportBuf_, s.c_str(), s.size());
                    blenderExportBuf_[s.size()] = '\0';
                }
                startBlenderExport(s);
            }

            // Small top-left overlay under the main menu bar showing FPS and visible count
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImGui::SetNextWindowPos(ImVec2(padding, y), ImGuiCond_Always);

                ImGui::Begin("StatsOverlay", nullptr, flags);

                // Statistics: loaded/visible counts
                ImGui::Text("Textures Loaded (CPU): %u", loadedTextureLayers);

                // Opaque (solid) — async snapshot taken in processPendingMeshes
                // (1-frame latency, never touches GPU memory on the UI path).
                size_t opaqueLoaded = sceneRenderer->solidRenderer->getIndirectRenderer().getMeshCount();
                uint32_t opaqueVisible = sceneRenderer ? sceneRenderer->getLastOpaqueVisible() : 0;
                ImGui::Text("Opaque - Loaded (GPU): %zu  Visible (GPU cull): %u", opaqueLoaded, opaqueVisible);
                size_t opaqueTracked = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;
                ImGui::Text("Opaque Models Tracked: %zu", opaqueTracked);

                // Transparent / water
                size_t transparentLoaded = sceneRenderer && sceneRenderer->waterRenderer ? sceneRenderer->waterRenderer->getIndirectRenderer().getMeshCount() : 0;
                uint32_t transparentVisible = sceneRenderer ? sceneRenderer->getLastTransparentVisible() : 0;
                ImGui::Text("Transparent - Loaded (GPU): %zu  Visible (GPU cull): %u", transparentLoaded, transparentVisible);
                size_t transparentTracked = sceneRenderer ? sceneRenderer->getTransparentModelCount() : 0;
                ImGui::Text("Transparent Models Tracked: %zu", transparentTracked);

                // Vegetation
                size_t vegChunks = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getChunkCount() : 0;
                size_t vegInstances = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getInstanceTotal() : 0;
                ImGui::Text("Vegetation Chunks: %zu", vegChunks);
                ImGui::Text("Vegetation Instances: %zu", vegInstances);

              
                if (profilingEnabled) {
                    ImGui::Separator();
                    ImGui::Text("--- GPU Timing (ms) ---");
                    float gpuTotal = profileDepthPrepass + profileSky + profileSolidDraw +
                                     profileVegetationImpostor + profileWater + profileRTDispatch +
                                     profileSdf + profilePostProcess + profileImGui;
                    ImGui::Text("Depth Prepass: %.2f", profileDepthPrepass);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("Solid Draw:    %.2f", profileSolidDraw);
                    ImGui::Text("Vegetation:    %.2f", profileVegetationImpostor);
                    ImGui::Text("Water:         %.2f", profileWater);
                    ImGui::Text("RT Dispatch:   %.2f", profileRTDispatch);
                    ImGui::Text("SDF Engine:    %.2f", profileSdf);
                    ImGui::Text("PostProcess:   %.2f", profilePostProcess);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("--- GPU Total:  %.2f ---", gpuTotal);
                    ImGui::Separator();
                    ImGui::Text("--- Hybrid RT ---");
                    if (sceneRenderer && sceneRenderer->rayTracing && sceneRenderer->rayTracing->isSupported()) {
                        ImGui::Text("Proxies: %u  Builds: %u (%.2f ms last)",
                            sceneRenderer->rayTracing->proxyCount(),
                            sceneRenderer->rayTracing->buildCount(),
                            sceneRenderer->rayTracing->lastBuildMs());
                        ImGui::Text("Pipeline: %s  TLAS: %s",
                            sceneRenderer->rayTracing->isPipelineReady() ? "ready" : "inline-only",
                            sceneRenderer->rayTracing->tlasBuilt() ? "built" : "pending");
                    } else {
                        ImGui::Text("RT unsupported — raster + CSM fallback");
                    }
                    // ── Per-op RT GPU profiling (opt-in) ──
                    // The instrumented shader variants accumulate per-op ray
                    // counts, hit counts and device-clock thread-time into a
                    // per-frame GPU buffer (1/4 of invocations sampled; scaled
                    // up here). Time is summed over shader invocations: the ops
                    // run concurrently, so compare the shares, not the absolute
                    // ms against frame time.
                    ImGui::Checkbox("RT op profiling", &rtProfilingEnabled_);
                    if (rtProfilingEnabled_) {
                        if (!rtProfilingSupported) {
                            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                                "shader clock / storage atomics unsupported - profiling off");
                        } else {
                            static const char* kRtOpNames[RTProfileCounters::kOpCount] = {
                                "Solid refl", "Water refl", "Water refr", "Water depth",
                                "Bounce/dbl", "Contact shdw", "Water-hit refr"
                            };
                            float timeMs[RTProfileCounters::kOpCount];
                            float totalMs = 0.0f;
                            for (uint32_t o = 0; o < RTProfileCounters::kOpCount; ++o) {
                                // Counter unit = 64 ns, 1/4 sampled => 256 ns each.
                                timeMs[o] = static_cast<float>(rtProfileStats_.time[o])
                                    * 256.0f * 1e-6f;
                                totalMs += timeMs[o];
                            }
                            ImGui::Text("--- RT Ops (thread-ms, %.1f total) ---", totalMs);
                            for (uint32_t o = 0; o < RTProfileCounters::kOpCount; ++o) {
                                const float raysK = static_cast<float>(rtProfileStats_.rays[o]) * 4.0f * 1e-3f;
                                const float hitsK = static_cast<float>(rtProfileStats_.hits[o]) * 4.0f * 1e-3f;
                                const float pct = (totalMs > 0.001f) ? (timeMs[o] / totalMs * 100.0f) : 0.0f;
                                ImGui::Text("%-12s %7.2f ms %5.0f%%  rays %6.0fK hits %6.0fK",
                                    kRtOpNames[o], timeMs[o], pct, raysK, hitsK);
                            }
                        }
                    }
                    // ── SDF engine (generic raymarch: fire/smoke) ──
                    ImGui::Separator();
                    ImGui::Text("--- SDF Engine ---");
                    if (sceneRenderer && sceneRenderer->sdfRenderer) {
                        auto* sdf = sceneRenderer->sdfRenderer.get();
                        const SdfStats& st = sdf->getStats();
                        ImGui::Text("GPU pass: %.2f ms   CPU task*: %.2f ms", profileSdf, profileSdfCpu);
                        ImGui::Text("containers: %u  defs: %u  mats: %u  cells: %u",
                            st.containerCount, st.definitionCount, st.materialCount, st.gridCellCount);
                        ImGui::Text("draw instances: %u (proxy cubes)", st.lastDrawInstances);
                        // March counters (opt-in fragment atomics; gated by the
                        // CPU-written enabled word so the off path stays free).
                        if (sdf->profileCountersSupported()) {
                            if (ImGui::Checkbox("SDF march counters", &sdfProfilingEnabled_)) {
                                sdf->setProfileCounters(sdfProfilingEnabled_);
                            }
                        } else {
                            ImGui::TextDisabled("march counters unsupported (fragmentStoresAndAtomics)");
                        }
                        if (sdfProfilingEnabled_) {
                            const double rays = static_cast<double>(st.lastFragments);
                            const double cells = static_cast<double>(st.lastCellVisits);
                            ImGui::Text("rays: %llu  steps: %llu  cells: %llu",
                                (unsigned long long)st.lastFragments,
                                (unsigned long long)st.lastMarchSteps,
                                (unsigned long long)st.lastCellVisits);
                            ImGui::Text("candidates: %llu  hits: %llu",
                                (unsigned long long)st.lastCandidateEvals,
                                (unsigned long long)st.lastHits);
                            ImGui::Text("avg steps/ray: %.1f  candidates/cell: %.1f  hit rate: %.1f%%",
                                rays > 0.0 ? static_cast<double>(st.lastMarchSteps) / rays : 0.0,
                                cells > 0.0 ? static_cast<double>(st.lastCandidateEvals) / cells : 0.0,
                                rays > 0.0 ? 100.0 * static_cast<double>(st.lastHits) / rays : 0.0);
                        }
                        ImGui::Text("lava: %u/%u chunks",
                            st.lavaAnchors, st.lavaChunks);
                    } else {
                        ImGui::Text("no SDF renderer");
                    }
                    ImGui::Separator();
                    ImGui::Text("--- CPU Timing (ms) ---");
                    ImGui::Text("Backface*:     %.2f", profileBackface);
                    ImGui::Text("* = CPU-timed (async)");
                    ImGui::Separator();
                    ImGui::Text("FPS:           %.1f", profileFps);
                    ImGui::Text("Update:        %.2f", profileCpuUpdate);
                    ImGui::Text("Record:        %.2f", profileCpuRecord);
                    // Queue submits/frame (perf report 22 C3): diffed cumulative
                    // counter; approximate under worker lag, exact enough to
                    // prove consolidation deltas.
                    {
                        static uint64_t lastSubmits = 0;
                        const uint64_t total = getTotalSubmitted();
                        const uint64_t perFrame = (total >= lastSubmits) ? (total - lastSubmits) : total;
                        lastSubmits = total;
                        ImGui::Text("Submits:       %llu", (unsigned long long)perFrame);
                    }
                }

                // GPU memory usage (VK_EXT_memory_budget)
                {
                    auto budgets = getMemoryBudgets();
                    if (!budgets.empty()) {
                        ImGui::Separator();
                        ImGui::Text("--- GPU Memory (MB) ---");
                        for (size_t h = 0; h < budgets.size(); ++h) {
                            const auto& b = budgets[h];
                            const char* heapName = (b.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "Device Local" : "Host Visible";
                            float usageMB  = static_cast<float>(b.usage)  / (1024.0f * 1024.0f);
                            float budgetMB = static_cast<float>(b.budget) / (1024.0f * 1024.0f);
                            float totalMB  = static_cast<float>(b.size)   / (1024.0f * 1024.0f);
                            if (b.budget > 0) {
                                ImGui::Text("%s:  %.0f / %.0f MB", heapName, usageMB, budgetMB);
                            } else {
                                ImGui::Text("%s:  %.0f / %.0f MB (total)", heapName, usageMB, totalMB);
                            }
                        }
                    }
                }

                ImGui::End();
            }

            // Small top-right overlay under the main menu bar showing gamepad connection
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImVec2 disp = ImGui::GetIO().DisplaySize;
                // anchor by top-right using pivot (1,0)
                ImGui::SetNextWindowPos(ImVec2(disp.x - padding, y), ImGuiCond_Always, ImVec2(1.0f, 0.0f));

                bool gamepadConnected = false;
                for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; ++jid) {
                    if (glfwJoystickIsGamepad(jid)) { gamepadConnected = true; break; }
                }
                bool wiimoteConnected = nunchukPublisher.isConnected();

                ImGui::Begin("GamepadOverlay", nullptr, flags);

                // Keyboard (always present)
                {
                    const char* catLabel = controllerManager.keyboardContext.activeCategory() == PageCategory::CAMERA ? "CAM" : "BRU";
                    ImGui::Text("Keyboard %s", catLabel);
                    ImGui::Text("%s", controllerManager.keyboardContext.activeSubpageName().c_str());
                }

                if (gamepadConnected) {
                    ImGui::Separator();
                    const char* catLabel = controllerManager.gamepadContext.activeCategory() == PageCategory::CAMERA ? "CAM" : "BRU";
                    ImGui::Text("Gamepad %s", catLabel);
                    ImGui::Text("%s", controllerManager.gamepadContext.activeSubpageName().c_str());
                }

                if (wiimoteConnected) {
                    ImGui::Separator();
                    const char* catLabel = controllerManager.wiimoteContext.activeCategory() == PageCategory::CAMERA ? "CAM" : "BRU";
                    ImGui::Text("Wiimote %s", catLabel);
                    ImGui::Text("%s", controllerManager.wiimoteContext.activeSubpageName().c_str());
                }

                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        // Update per-frame widget state (avoid storing VulkanApp* inside widgets)
        if (renderTargetsWidget) renderTargetsWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());
        if (vulkanResourcesManagerWidget) vulkanResourcesManagerWidget->updateWithApp(this);
        if (queueTimelineWidget) queueTimelineWidget->updateWithApp(this);

        // Render radial menu (behind all widgets)
        if (radialMenu) {
            radialMenu->Update();
            radialMenu->Draw();
        }

        // Blender export: progress while running, result modal when finished.
        if (blenderExportRunning_.load()) {
            std::string phase;
            {
                std::lock_guard<std::mutex> lock(blenderExportMutex_);
                phase = blenderExportPhase_;
            }
            if (ImGui::Begin("Blender Export", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextUnformatted("Tessellating the octree and writing the .glb...");
                ImGui::ProgressBar(blenderExportProgress_.load(), ImVec2(360.0f, 0.0f));
                ImGui::TextDisabled("%s", phase.c_str());
                if (ImGui::Button("Cancel")) blenderExportCancel_.store(true);
            }
            ImGui::End();
        }
        if (blenderExportStatusPending_) {
            ImGui::OpenPopup("Blender Export Result");
            blenderExportStatusPending_ = false;
        }
        if (ImGui::BeginPopupModal("Blender Export Result", nullptr,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            blender_export::Stats stats;
            {
                std::lock_guard<std::mutex> lock(blenderExportMutex_);
                stats = blenderExportStats_;
            }
            if (stats.ok) {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f),
                                   stats.blendWritten ? "Blender scene exported"
                                                      : "Export complete");
                ImGui::Text("%zu chunks, %zu solid tris, %zu water tris",
                            stats.chunks, stats.solidTriangles, stats.waterTriangles);
                ImGui::Text("%zu grass tufts, %zu SDF instances, %zu materials",
                            stats.grassInstances, stats.sdfInstances, stats.usedMaterials);
                ImGui::Separator();
                if (stats.blendWritten) {
                    ImGui::TextDisabled("%s", stats.blendPath.c_str());
                }
                if (!stats.outputPath.empty()) {
                    ImGui::TextDisabled("%s", stats.outputPath.c_str());
                }
                ImGui::TextDisabled("%s", stats.scriptPath.c_str());
                ImGui::TextDisabled("%s", stats.configPath.c_str());
                ImGui::TextDisabled("%s", stats.readmePath.c_str());
                if (!stats.blenderLogPath.empty()) {
                    ImGui::TextDisabled("%s", stats.blenderLogPath.c_str());
                }
                if (!stats.warning.empty()) {
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.1f, 1.0f), "%s",
                                       stats.warning.c_str());
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Export failed");
                ImGui::TextWrapped("%s", stats.error.c_str());
            }
            ImGui::Separator();
            if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        // Render all widgets
        widgetManager.renderAll();

        // Music analysis runs even when the widget window is hidden, then
        // the compact features are forwarded into the water GPU params
        // (SSBO musicAudio vecs) for MUSIC_REACTIVE_WATER. The analyser owns
        // the FFT; the shader only reads these uploaded values.
        if (mp3Widget) {
            mp3Widget->tick();
            if (sceneRenderer && sceneRenderer->waterRenderer) {
                const AudioFeatures& af = mp3Widget->audioFeatures();
                const bool active = mp3Widget->musicReactiveActive() && af.valid;
                sceneRenderer->waterRenderer->updateMusicAudio(
                    af.amplitude, af.bassEnergy, af.midEnergy, af.highEnergy,
                    af.beatIntensity, active);
            }
        }
    }

    void draw(VkCommandBuffer &commandBuffer) override {
        // Only record draw commands; command buffer and render pass are already active
        if (commandBuffer == VK_NULL_HANDLE) {
            std::cerr << "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE, skipping draw." << std::endl;
            return;
        }
        if (!sceneRenderer) {
            std::cerr << "[MyApp::draw] Error: sceneRenderer is nullptr, skipping draw." << std::endl;
            return;
        }
        if (!world) {
            std::cerr << "[MyApp::draw] Error: world is nullptr, skipping draw." << std::endl;
            return;
        }

    

        uint32_t frameIdx = getCurrentFrame();

        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Composite offscreen scene + water + brush into the swapchain
        if (sceneRenderer && sceneRenderer->postProcessRenderer) {
            VkImageView skyViewPP = sceneRenderer->skyRenderer ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            VkImageView brushColorView = sceneRenderer->brushRenderer ? sceneRenderer->brushRenderer->getColorView(frameIdx) : VK_NULL_HANDLE;
            VkImageView brushDepthView = sceneRenderer->brushRenderer ? sceneRenderer->brushRenderer->getDepthView(frameIdx) : VK_NULL_HANDLE;
            VkImageView waterGeomDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->waterRenderer) {
                waterGeomDepthView = sceneRenderer->waterRenderer->getWaterGeomDepthView(frameIdx);
            }
            VkImageView vegColorView = VK_NULL_HANDLE;
            VkImageView vegDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->vegetationRenderer) {
                vegColorView = sceneRenderer->vegetationRenderer->getVegColorView(frameIdx);
                vegDepthView = sceneRenderer->vegetationRenderer->getVegDepthView(frameIdx);
            }
            VkImageView sdfColorView = VK_NULL_HANDLE;
            VkImageView sdfDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->debugSDFRenderer) {
                sdfColorView = sceneRenderer->debugSDFRenderer->getSdfColorView(frameIdx);
                sdfDepthView = sceneRenderer->debugSDFRenderer->getSdfDepthView(frameIdx);
            }
            VkImageView bboxColorView = VK_NULL_HANDLE;
            VkImageView bboxDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->boundingBoxRenderer) {
                bboxColorView = sceneRenderer->boundingBoxRenderer->getBboxColorView(frameIdx);
                bboxDepthView = sceneRenderer->boundingBoxRenderer->getBboxDepthView(frameIdx);
            }
            VkImageView fireColorView = VK_NULL_HANDLE;
            VkImageView fireDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->sdfRenderer) {
                fireColorView = sceneRenderer->sdfRenderer->getSdfColorView(frameIdx);
                fireDepthView = sceneRenderer->sdfRenderer->getSdfDepthView(frameIdx);
            }
            VkImageView videoColorView = sceneRenderer->videoRenderer
                                             ? sceneRenderer->videoRenderer->colorView(frameIdx) : VK_NULL_HANDLE;
            VkImageView videoDepthView = sceneRenderer->videoRenderer
                                             ? sceneRenderer->videoRenderer->depthView(frameIdx) : VK_NULL_HANDLE;
            float brushAlpha = 0.5f;
            float brushMode = 0.0f;
            const BrushEntry* brushEntry = brushManager.getSelectedEntry();
            if (brushEntry) {
                brushAlpha = brushEntry->opacity;
                brushMode = static_cast<float>(brushEntry->brushMode);
            }
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 16);
            sceneRenderer->postProcessRenderer->render(
                this,
                commandBuffer,
                sceneRenderer->solidRenderer->getColorView(frameIdx),
                sceneRenderer->solidRenderer->getDepthView(frameIdx),
                settings.waterInMainPass
                    ? sceneRenderer->waterRenderer->getDummyWaterColorView()
                    : sceneRenderer->waterRenderer->getWaterDepthView(frameIdx),
                // Water refraction+tint body (RGB+weight) and measured depth
                // for the final-pass depth-guided water blur (water-in-main
                // has no aux targets). L15: lazily unallocated body/column
                // targets bind the 1x1 dummy (descriptors must stay valid
                // even though the blur-gated composite never samples them).
                settings.waterInMainPass
                    ? sceneRenderer->waterRenderer->getDummyWaterColorView()
                    : (sceneRenderer->waterRenderer->getWaterBodyView(frameIdx) != VK_NULL_HANDLE
                       ? sceneRenderer->waterRenderer->getWaterBodyView(frameIdx)
                       : sceneRenderer->waterRenderer->getDummyWaterColorView()),
                settings.waterInMainPass
                    ? sceneRenderer->waterRenderer->getDummyWaterColorView()
                    : (sceneRenderer->waterRenderer->getWaterColumnView(frameIdx) != VK_NULL_HANDLE
                       ? sceneRenderer->waterRenderer->getWaterColumnView(frameIdx)
                       : sceneRenderer->waterRenderer->getDummyWaterColorView()),
                brushColorView,
                brushDepthView,
                waterGeomDepthView,
                vegColorView,
                vegDepthView,
                sdfColorView,
                sdfDepthView,
                bboxColorView,
                bboxDepthView,
                fireColorView,
                fireDepthView,
                videoColorView,
                videoDepthView,
                brushAlpha,
                brushMode,
                viewProj,
                invViewProj,
                uboStatic.viewPosition,
                frameIdx,
                skyViewPP,
                // Body/column blur gate (perf_report_19 H4): when no layer
                // needs the final-pass blur the geometry pass skips the aux
                // targets and the composite must not fetch them. water-in-main
                // has no aux targets at all.
                !settings.waterInMainPass
                    && sceneRenderer->waterRenderer->waterBlurNeeded(),
                // M12 (perf report 22): the vegetation offscreen targets are
                // downscaled -> composite takes the closest of the 2x2
                // veg-depth taps instead of one bilinear sample.
                settings.vegetationRenderScale < 0.999f,
                // Shared wind field (binding 27) + clock for the wind debug
                // raymarch overlay. Buffer owned/streamed by
                // VegetationRenderer; mainTime matches the vegetation clock.
                sceneRenderer->vegetationRenderer->getWindFieldBuffer().buffer,
                mainTime);
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 17);
        }

        // Brush overlay is composited inside the PostProcess pass using the
        // brush_color and brush_depth targets from the early brush pass, with
        // depth testing against both scene_depth (solids) and water_depth.

        // ImGui rendering
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (!draw_data) {
            std::cerr << "[MyApp::draw] Warning: ImGui::GetDrawData() returned nullptr, skipping ImGui rendering." << std::endl;
        } else if (commandBuffer == VK_NULL_HANDLE) {
            std::cerr << "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE before ImGui rendering, skipping ImGui." << std::endl;
        } else {
            VkQueryPool qp = queryPools[getCurrentFrame()];
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 18);
            {
                // ImGui's dynamic texture uploads (font atlas created on the
                // first frame, widget textures) issue their own vkQueueSubmit /
                // vkQueueWaitIdle on this queue. Serialize with the engine's
                // frame and async upload submissions.
                std::lock_guard<std::mutex> qlock(getQueueSubmitMutex(getGraphicsQueue()));
                ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
            }
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 19);
        }
    }

    void clean() override {
    // Stop the Blender export worker before any scene/resource teardown: it
    // holds the scene alive through world->scene().
    blenderExportCancel_.store(true);
    if (blenderExportThread_.joinable()) blenderExportThread_.join();
    // Exit viewpoint, logged next to the saved exit screenshot so a report
    // of "X is not visible" can be reproduced pixel-exact in the CPU probe
    // (same format as the startup [Camera Setup] line).
    printf("[Camera Exit] Position: (%.1f, %.1f, %.1f)\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
    printf("[Camera Exit] Forward: (%.3f, %.3f, %.3f)\n", camera.getForward().x, camera.getForward().y, camera.getForward().z);
    // Ensure all GPU work is finished before tearing down any Vulkan resources.
    // Belt-and-suspenders: VulkanApp::cleanup() already calls deviceWaitIdle()
    // before clean(), but draining the upload queues and stopping thread pools
    // may have left work in flight.
    deviceWaitIdle();

    // Video decoding workers must stop before the device (and the array layers
    // they upload into) go away. Deferred upload callbacks hold a weak reference
    // to the service internals, so they become no-ops after shutdown().
    if (videoTextureService) {
        videoTextureService->shutdown();
        videoTextureService.reset();
    }

    // Remote streaming stops first: disconnect wakes mesh fetchers blocked on
    // the socket (they run on the gen pools stopped below), then joins the
    // receiver and dispatch threads.
    stopRemoteStreaming();

    // Join tessellation thread before tearing down Vulkan resources.
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

        // Free all ImGui descriptor sets owned by the widget while ImGui is still
        // alive. clean() is called before cleanupImGui(), so this is safe.
        // Without this, ~RenderTargetsWidget() (called from ~MyApp() after Vulkan
        // teardown) would try to free descriptors from a destroyed pool.
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }
        if (videoTextureService) {
            videoTextureService->invalidatePreviews();
        }

        if (impostorService) {
            impostorService->invalidateImGuiDescriptors();
            impostorService->cleanup();
        }

        // Stop serving before the scene is torn down (the providers reference
        // it, so the transport must be closed first).
        if (sceneServer) {
            sceneServer->stop();
            sceneServer.reset();
        }

        // Droplet sim/renderer own device-local buffers created through the
        // app; destroy them before the scene renderer drains its own resources.
        if (particleSim) {
            particleSim->destroy(this);
            particleSim.reset();
        }
        if (dropletRenderer) {
            dropletRenderer->destroy(this);
            dropletRenderer.reset();
        }

        // Stop ALL thread pools BEFORE sceneRenderer->cleanup() drains the
        // upload queues. Workers may push new upload jobs via tessellation
        // callbacks; if pools are still running when streamer.destroy()
        // drains the MPSCQueues, new jobs pushed after the drain will be
        // left as orphans and crash during ~MPSCQueue().
        asyncThreadPool.stop();
        if (world) world->stopPools();
        if (sceneRenderer) sceneRenderer->stopGenPools();

        // Cleanup scene renderer and all sub-renderers (must happen while
        // the Vulkan device is still alive). streamer.destroy() drains the
        // upload queues — safe now because all pools are stopped.
        if (sceneRenderer) {
            sceneRenderer->cleanup(this);
        }

        // Destroy timestamp query pools
        for (uint32_t f = 0; f < 3; ++f) {
            if (queryPools[f] != VK_NULL_HANDLE) {
                vkDestroyQueryPool(getDevice(), queryPools[f], nullptr);
                queryPools[f] = VK_NULL_HANDLE;
            }
        }
        // Pre-allocated ring pools are tracked by VulkanResourceManager, so
        // resources.cleanup() will destroy them. Just zero our arrays.
        for (auto& slot : cachedBackfaceRing) slot = {};

        // NOTE: Vulkan-owned objects for global managers are now cleaned up by
        // `VulkanResourceManager::cleanup(device)`. Avoid calling manager-level
        // destroy/cleanup routines here that perform Vulkan destroys to prevent
        // double-destruction ordering issues. If a manager needs CPU-only
        // cleanup, add a dedicated method and call it here.

        // Delete scene objects while the Vulkan device is still alive. Their
        // destructors (TerrainStreamer → UploadManager → MPSCQueue) must not
        // run after vkDestroyDevice().
        delete sceneRenderer; sceneRenderer = nullptr;
        delete world; world = nullptr;
    }

    void stopBackgroundThreads() override {
        // Device-lost teardown path: join/stop CPU threads only, never touch
        // Vulkan (any vkDestroy* on objects still tracked in use would trip a
        // validation error). Mirrors the thread-stopping half of clean().
        // Remote streaming stops first so blocked mesh fetchers wake up.
        stopRemoteStreaming();
        if (sceneProcessThread.joinable()) sceneProcessThread.join();
        asyncThreadPool.stop();
        if (world) world->stopPools();
        if (sceneRenderer) sceneRenderer->stopGenPools();
    }

    void onSwapchainResized(uint32_t width, uint32_t height) override {
        if (sceneRenderer) {
            sceneRenderer->onSwapchainResized(this, width, height);
        }
        // H8: the sky offscreen targets were recreated with undefined contents
        // (and possibly new handles); force re-rendering until the cache is
        // repopulated. A few warmup frames cover all in-flight consumers.
        {
            std::lock_guard<std::mutex> lock(skySigMutex);
            skyRenderRuns = 0;
            skySigValid = false;
        }
    }

    void preImGuiShutdown() override {
        // Free all ImGui descriptor sets BEFORE Shutdown while the old backend data
        // is still alive. This avoids freeing DS allocated with the old descriptor
        // set layout through the new backend data after a Shutdown/Init cycle.
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            sceneRenderer->shadowMapper->freeImGuiDescriptors();
        }
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }
        textureArrayManager.invalidateImGuiDescriptors();
        if (videoTextureService) {
            videoTextureService->invalidatePreviews();
        }
        if (impostorService) {
            impostorService->invalidateImGuiDescriptors();
        }
    }

    void onImGuiRecreated() override {
        // Re-create ImGui AddTexture DS for shadow cascades — the old ones used the
        // previous DescriptorSetLayout which was destroyed by ImGui_ImplVulkan_Shutdown.
        // Old handles were freed by preImGuiShutdown() so recreateImGuiDescriptors
        // will skip the free and go straight to allocation.
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            sceneRenderer->shadowMapper->recreateImGuiDescriptors();
        }
        // Widget handles were nulled by preImGuiShutdown(); this is a no-op.
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }
        if (videoTextureService) {
            videoTextureService->invalidatePreviews();
        }
        if (impostorService) {
            impostorService->recreateImGuiDescriptors();
        }
    }

    void onEvent(const EventPtr &event) override {
        if (auto closeEvent = std::dynamic_pointer_cast<CloseWindowEvent>(event)) {
            requestClose();
            return;
        }
        if (auto fullscreenEvent = std::dynamic_pointer_cast<ToggleFullscreenEvent>(event)) {
            toggleFullscreen();
            return;
        }
        if (auto qualityEvent = std::dynamic_pointer_cast<SetGraphicsQualityEvent>(event)) {
            // Runs on the queued-event drain (main thread, before frame
            // recording). The preset edits the global Settings gates only;
            // the authored per-layer WaterSettings are left untouched, so
            // nothing has to be re-uploaded to the water GPU params.
            GraphicsSettingsCommand command(qualityEvent->quality);
            command.execute(settings);
            return;
        }
        if (auto rebuildEvent = std::dynamic_pointer_cast<RebuildBrushEvent>(event)) {
            // The preview is streamed per frame from the selected entry, so a
            // rebuild event needs no deferred work; kept for compatibility
            // with publishers that announce brush edits.
            (void)rebuildEvent;
            return;
        }
        if (auto applyEvent = std::dynamic_pointer_cast<ApplyBrushToSceneEvent>(event)) {
            brushApplyToScenePending = true;
            return;
        }
        if (auto texEvent = std::dynamic_pointer_cast<SetBrushTextureEvent>(event)) {
            BrushEntry* be = brushManager.getSelectedEntry();
            if (be && be->materialIndex != texEvent->index) {
                be->materialIndex = texEvent->index;
            }
            return;
        }
        if (auto sdfEvent = std::dynamic_pointer_cast<SetBrushSdfTypeEvent>(event)) {
            BrushEntry* be = brushManager.getSelectedEntry();
            if (be && be->sdfType != sdfEvent->sdfType) {
                be->sdfType = sdfEvent->sdfType;
                eventManager.queue(std::make_shared<RebuildBrushEvent>());
            }
            return;
        }
        if (auto ctrlEvent = std::dynamic_pointer_cast<SetBrushControlEvent>(event)) {
            brushManager.controlMode = ctrlEvent->mode;

            // Map BrushControlMode to PageControl
            PageControl pc = PageControl::TRANSLATE;
            switch (ctrlEvent->mode) {
                case BrushControlMode::TRANSLATE:  pc = PageControl::TRANSLATE;  break;
                case BrushControlMode::AIM:        pc = PageControl::AIM;        break;
                case BrushControlMode::SCALE:      pc = PageControl::SCALE;      break;
                case BrushControlMode::TEXTURE:    pc = PageControl::TEXTURE;    break;
                case BrushControlMode::ATTRIBUTE:  pc = PageControl::ATTRIBUTE;  break;
                case BrushControlMode::COLOR:      pc = PageControl::COLOR;      break;
                default: break;
            }

            // Switch all controller contexts to Brush page + selected subpage
            controllerManager.switchAllToBrush(pc);
            return;
        }
        if (auto pageEvent = std::dynamic_pointer_cast<SetPageEvent>(event)) {
            brushManager.controlMode = pageEvent->brushMode;

            controllerManager.switchAllContexts(pageEvent->category, pageEvent->control);
            return;
        }
        if (auto paintEvent = std::dynamic_pointer_cast<SetBrushPaintModeEvent>(event)) {
            brushManager.paintMode = paintEvent->mode;

            // Update brushMode on the selected entry so the renderer sees it
            BrushEntry* be = brushManager.getSelectedEntry();
            if (be) {
                switch (paintEvent->mode) {
                    case BrushPaintMode::ADD:    be->brushMode = 0; break;
                    case BrushPaintMode::REMOVE: be->brushMode = 1; break;
                    case BrushPaintMode::PAINT:  be->brushMode = 2; break;
                    default: break;
                }
                eventManager.queue(std::make_shared<RebuildBrushEvent>());
            }

            controllerManager.switchAllToBrush(PageControl::ATTRIBUTE);
            return;
        }
        if (auto dragEvent = std::dynamic_pointer_cast<SetBrushDragModeEvent>(event)) {
            brushManager.dragMode = dragEvent->mode;

            controllerManager.switchAllToBrush(PageControl::ATTRIBUTE);
            return;
        }
        if (auto hsvEvent = std::dynamic_pointer_cast<SetBrushHSVEvent>(event)) {
            BrushEntry* be = brushManager.getSelectedEntry();
            if (be) {
                if (hsvEvent->component == "Hue") {
                    be->hsv.x = hsvEvent->value;
                } else if (hsvEvent->component == "Saturation") {
                    be->hsv.y = hsvEvent->value / 100.0f;
                } else if (hsvEvent->component == "Value") {
                    be->hsv.z = hsvEvent->value / 100.0f;
                }
                eventManager.queue(std::make_shared<RebuildBrushEvent>());
            }
            return;
        }
        if (auto lightEvent = std::dynamic_pointer_cast<SetLightEvent>(event)) {
            float azi, ele;
            light.getSpherical(azi, ele);
            if (lightEvent->component == "Azimuth") {
                light.setFromSpherical(lightEvent->value - 180.0f, ele);
            } else if (lightEvent->component == "Elevation") {
                light.setFromSpherical(azi, lightEvent->value - 90.0f);
            }
            return;
        }
    }

    // Called by VulkanApp after a frame has been submitted
    void postSubmit() override;
};



int main(int argc, char** argv) {
    // Allow threads (RADV driver, validation layers, miniaudio) to use
    // real-time scheduling. Without this, glibc's thread priority protection
    // code (tpp.c) hits an assertion when a library tries to promote a thread
    // to SCHED_FIFO but RLIMIT_RTPRIO is too low.
    struct rlimit rlim;
    if (getrlimit(RLIMIT_RTPRIO, &rlim) == 0) {
        rlim.rlim_cur = rlim.rlim_max;
        setrlimit(RLIMIT_RTPRIO, &rlim);
    }
    // Remote scene endpoint: --remote=HOST:PORT (or --remote HOST:PORT),
    // --local to skip the connection attempt, env VULKAN_REMOTE likewise
    // ("off"/"0" disables). Default tries 127.0.0.1:8080 and falls back to
    // LocalScene when unreachable.
    std::string remoteEndpoint = "127.0.0.1:8080";
    bool remoteDisabled = false;
    if (const char* env = std::getenv("VULKAN_REMOTE")) {
        const std::string e = env;
        if (e.empty() || e == "off" || e == "0") remoteDisabled = true;
        else remoteEndpoint = e;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--local") remoteDisabled = true;
        else if (a.rfind("--remote=", 0) == 0) remoteEndpoint = a.substr(9);
        else if (a == "--remote" && i + 1 < argc) remoteEndpoint = argv[++i];
    }
    try {
        MyApp app;
        app.remoteEndpoint = remoteEndpoint;
        app.remoteDisabled = remoteDisabled;
        app.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

// Perf report 23 C1: rebuild the five material arrays at a new resolution.
// Called from preRenderPass when Settings::textureArraySize changes, so the
// frame is not yet recorded and the device can be idled for the rebuild.
void MyApp::recreateTextureArrays(uint32_t size) {
    if (textureArrayManager.layerAmount == 0) return;
    if (textureArrayManager.width == size) return;
    const uint32_t layers = textureArrayManager.layerAmount;
    const uint32_t oldSize = textureArrayManager.width;
    std::cout << "[MyApp] texture arrays: " << oldSize << "^2 -> " << size
              << "^2 x " << layers << " layers (perf report 23 C1)\n";

    // Major resource rebuild: the five arrays are sampled by the solid pass
    // (graphics queue), the brush/vegetation passes and the composite, and
    // written by the mixer. This is the AGENTS.md "major resource rebuild"
    // case where a device idle is allowed; it also makes the deferred
    // destroys below provably runnable through processPendingCommandBuffers().
    vkDeviceWaitIdle(getDevice());
    processPendingCommandBuffers();

    // Swap the images underneath the manager (deferred-frees the old ones,
    // creates the new, bumps version, notifies the allocation listeners once
    // with the new views), then re-upload the same triples.
    textureArrayManager.recreate(this, layers, size, size);
    std::vector<TextureTriple> textureTripleViews;
    textureTripleViews.reserve(textureTriples_.size());
    for (const SceneTextureTriple& t : textureTriples_) {
        textureTripleViews.push_back({t.albedo.c_str(), t.normal.c_str(), t.height.c_str(),
                                      t.rough.c_str(), t.ao.c_str()});
    }
    loadedTextureLayers = textureArrayManager.loadTriples(this, textureTripleViews);
    if (loadedTextureLayers != textureTriples_.size()) {
        std::cerr << "[MyApp] texture-array rebuild loaded " << loadedTextureLayers
                  << " of " << textureTriples_.size() << " triples (capacity "
                  << textureArrayManager.layerAmount << " layers)\n";
    }

    // Refresh the five generated layers. The mixer rebuilds its compute-view
    // bindings per generation (perf report 23 C2), so the re-created arrays
    // need no explicit rebind here.
    if (textureMixer) {
        textureMixer->generateInitialTextures(mixerParams);
    }

    // Re-flag the editable placeholder layer (allocate() reset the flags).
    // loadTriples() only returns the triple count, so the generated mixer
    // layers are counted explicitly (mixerParams holds one entry per generated
    // layer) — otherwise loadedTextureLayers would shrink to the triple count
    // after every rebuild.
    const uint32_t editableLayer = static_cast<uint32_t>(textureTriples_.size() + mixerParams.size());
    loadedTextureLayers = std::min(editableLayer, layers);
    if (editableLayer < layers) {
        textureArrayManager.setLayerInitialized(editableLayer, true);
    }
    // The video array has its own size/lifetime and survives the rebuild; only
    // the conversion push constants and descriptor bindings are refreshed.
    if (videoTextureService) videoTextureService->onArraysRecreated();

    // The old images/samplers/views are deferred; every queue is idle, so this
    // runs the callbacks now and the transient double commitment is one call
    // long. Must come after the mixer rebind so no bound set references a
    // freed sampler.
    processPendingCommandBuffers();

    textureArrayManager.logMemoryUtilization("main");
}

// Implementation: setup vegetation textures
void MyApp::setupVegetationTextures() {
    // Bladed grass: no billboard textures. The impostor capture renders
    // procedural blade tufts directly; the capture arrays feed the far-LOD
    // impostor quads. Kept under the legacy name so setup() needs no change.
    impostorService = std::make_shared<ImpostorService>();
    // Must be set before init(): the capture pipeline shares the renderer's
    // set=2 wind params descriptor set/layout instead of duplicating them.
    impostorService->setVegetationRenderer(sceneRenderer->vegetationRenderer.get());
    impostorService->init(this);
    impostorWidget = std::make_shared<ImpostorWidget>(impostorService);
    impostorWidget->setVegetationRenderer(sceneRenderer->vegetationRenderer.get());

    // Capture the 3 blade variants x 20 views immediately (procedural, no
    // source textures) and wire the arrays into the vegetation renderer.
    if (impostorService) {
        impostorService->setSource(VK_NULL_HANDLE, VK_NULL_HANDLE,
                                   VK_NULL_HANDLE, VK_NULL_HANDLE);
    }
}

// Parsed-SVG cache for the SVG brush (sdfType 10). Keyed by the path as typed
// in the widget; failed loads are cached as null so a broken file logs once.
static std::shared_ptr<const SvgShape> loadSvgShapeCached(const std::string& path,
                                                          std::string* error = nullptr) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<const SvgShape>> cache;
    static std::unordered_map<std::string, std::string> errors;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = cache.find(path);
    if (it != cache.end()) {
        if (error) *error = errors[path];
        return it->second;
    }
    SvgShape shape;
    std::string err;
    std::string resolved;
    if (!SvgShape::loadFromFileResolved(path, shape, err, &resolved)) {
        errors[path] = err;
        cache[path] = nullptr;
        if (error) *error = err;
        std::cerr << "[SvgBrush] " << err << std::endl;
        return nullptr;
    }
    auto shared = std::make_shared<const SvgShape>(std::move(shape));
    std::cout << "[SvgBrush] loaded '" << resolved << "': " << shared->contours.size()
              << " contours" << std::endl;
    errors[path].clear();
    cache[path] = shared;
    if (error) error->clear();
    return shared;
}

// Construct a brush primitive for both the current transform (fn2) and the
// sweep start (fn1), wrap the pair in SweepSignedDistanceFunction when
// sweepMode is on, then run callback on the resulting function. `args` are
// the primitive's leading constructor arguments (radii/endpoints); model and
// minSize are appended for fn2, prevModel and minSize for fn1, matching the
// per-case construction order the switch used to spell out.
template<typename T, typename Fn, typename... Args>
static void applyPrimitive(const BrushEntry& entry, const Transformation& model,
                           const glm::vec3& sweepStart, float minSize,
                           Fn&& callback, const Args&... args) {
    T fn2(args..., model, minSize);
    if (entry.sweepMode) {
        Transformation prevModel(entry.scale, sweepStart, entry.rot);
        T fn1(args..., prevModel, minSize);
        SweepSignedDistanceFunction<T> sweepFn(fn1, fn2, model, minSize);
        callback(sweepFn);
    } else { callback(fn2); }
}

// Shared SDF creation: dispatches on entry.sdfType and calls callback on the
// constructed primitive (or its sweep wrapper).
template<typename Fn>
static void forEachBrushSDF(const BrushEntry& entry, const Transformation& model,
                            const glm::vec3& sweepStart, float minSize,
                            const char* logPrefix, Fn&& callback) {
    switch (entry.sdfType) {
        case 0: // Sphere
            applyPrimitive<SphereDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 1: // Box
            applyPrimitive<BoxDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 2: // Capsule
            applyPrimitive<CapsuleDistanceFunction>(entry, model, sweepStart, minSize, callback,
                entry.capsuleA, entry.capsuleB, entry.capsuleRadius);
            break;
        case 3: // Octahedron
            applyPrimitive<OctahedronDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 4: // Pyramid
            applyPrimitive<PyramidDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 5: // Torus
            applyPrimitive<TorusDistanceFunction>(entry, model, sweepStart, minSize, callback,
                entry.torusRadii);
            break;
        case 6: // Cone
            applyPrimitive<ConeDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 7: // Cylinder
            applyPrimitive<CylinderDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        case 8: // Tapered Cylinder
            applyPrimitive<TaperedCylinderDistanceFunction>(entry, model, sweepStart, minSize, callback,
                entry.taperedCylinderRadii.x, entry.taperedCylinderRadii.y);
            break;
        case 9: // Tapered Capsule
            applyPrimitive<TaperedCapsuleDistanceFunction>(entry, model, sweepStart, minSize, callback,
                entry.capsuleA, entry.capsuleB, entry.taperedCapsuleRadii.x, entry.taperedCapsuleRadii.y);
            break;
        case 10: { // SVG extrusion intersected with the brush box
            std::string svgError;
            std::shared_ptr<const SvgShape> shape = loadSvgShapeCached(entry.svgPath, &svgError);
            if (!shape) {
                std::cerr << logPrefix << " SVG brush (" << entry.svgPath << "): " << svgError
                          << std::endl;
                break;
            }
            applyPrimitive<SvgPolygonDistanceFunction>(entry, model, sweepStart, minSize,
                                                       callback, shape);
            break;
        }
        default:
            std::cerr << logPrefix << " Unknown sdfType " << entry.sdfType << ", skipping" << std::endl;
            break;
    }
}

// Shared effect+apply: wraps func in the selected effect (if any) and applies to octree.
static void applyBrushWithEffect(const BrushEntry& entry, SignedDistanceFunction& func,
                                 Octree& octree, const SignedDistanceOperation& op,
                                 const Transformation& model, const TexturePositionPainter& brush,
                                 const Simplifier& simplifier,
                                 Octree::OctreeNodeDataHandler& updateHandler,
                                 Octree::OctreeNodeDataHandler& deleteHandler) {
    float minSize = entry.minSize;
    if (entry.useEffect) {
        switch (entry.effectType) {
            case 0: {
                PerlinDistortDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 1: {
                PerlinCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency, entry.effectThreshold,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 2: {
                SineDistortDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency, glm::vec3(0), model, minSize);
                octree.apply(op, effect, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 3: {
                VoronoiCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectCellSize,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            default:
                octree.apply(op, func, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
        }
    } else {
        octree.apply(op, func, model, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
    }
}

// Bounds of one brush application (union of the current and, in sweep mode,
// the previous placement), padded for the optional effect displacement. Shared
// by the GPU preview and the undo history intersection tests.
static void computeBrushBounds(const BrushEntry& entry, const glm::vec3& sweepStart,
                               glm::vec3& boundsCenter, float& boundsRadius) {
    BoundingSphere bounds = [&]() {
        // Fallback for a brush whose primitive reports no sphere.
        BoundingSphere s;
        s.center = entry.translate;
        s.radius = glm::length(entry.scale);
        BrushEntry noSweep = entry;
        noSweep.sweepMode = false;
        auto sphereFor = [&](const glm::vec3& where) {
            Transformation m(entry.scale, where, entry.rot);
            forEachBrushSDF(noSweep, m, where, entry.minSize, "[computeBrushBounds]",
                [&](SignedDistanceFunction& func) {
                    BoundingSphere bs = func.getSphere(m, entry.minSize);
                    if (bs.radius > 0.0f) s = bs;
                    else { s.center = func.getCenter(); s.radius = glm::length(entry.scale); }
                });
        };
        sphereFor(entry.translate);
        if (entry.sweepMode) {
            const BoundingSphere current = s;
            sphereFor(sweepStart);
            const BoundingSphere prev = s;
            const glm::vec3 delta = current.center - prev.center;
            const float dist = glm::length(delta);
            const float radius = (dist + current.radius + prev.radius) * 0.5f;
            s.center = (dist < 1e-6f) ? current.center
                : prev.center + delta * ((radius - prev.radius) / dist);
            s.radius = radius;
        }
        return s;
    }();
    if (entry.useEffect) {
        float pad = entry.effectAmplitude * 1.97f;
        if (entry.effectType == 2) pad = entry.effectAmplitude * 0.5f;
        else if (entry.effectType == 3) pad = entry.effectAmplitude;
        bounds.radius += pad;
    }
    boundsRadius = std::max(bounds.radius, 0.1f) * 1.05f;
    boundsCenter = bounds.center;
}

// Per-frame brush preview update: stream the selected entry's SDF
// description into the BrushSdfRenderer's UBO (consumed by the preview
// raymarcher and by the solid shader's PAINT/REMOVE intersection test). No
// octree and no tessellation: moving/scaling the brush is just a UBO rewrite.
void MyApp::updateBrushPreview(uint32_t frameIndex) {
    if (!sceneRenderer || !sceneRenderer->brushRenderer) return;

    BrushEntry* entry = brushManager.getSelectedEntry();
    if (!entry) {
        // No brush: zero-radius bounds disable both the preview and the
        // solid PAINT/REMOVE evaluation.
        sceneRenderer->brushRenderer->updateParams(frameIndex, BrushEntry{}, glm::vec3(0.0f),
                                                   glm::vec3(0.0f), 0.0f,
                                                   getWidth(), getHeight());
        return;
    }

    // Sweep start = the position captured on the previous frame (the same pair
    // applyBrushToScene consumes via cachedSweepStart).
    // Diagnostic transform override (headless composite validation): forced
    // every frame because loading a scene replaces the manager's entries.
    // VULKAN_BRUSH_SCALE (uniform) / VULKAN_BRUSH_TRANSLATE=x,y,z.
    {
        static const char* scaleEnv = std::getenv("VULKAN_BRUSH_SCALE");
        static const char* translateEnv = std::getenv("VULKAN_BRUSH_TRANSLATE");
        if (scaleEnv != nullptr) {
            entry->scale = glm::vec3(std::max(1.0f, static_cast<float>(std::atof(scaleEnv))));
        }
        if (translateEnv != nullptr) {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (std::sscanf(translateEnv, "%f,%f,%f", &x, &y, &z) == 3) {
                entry->translate = glm::vec3(x, y, z);
            }
        }
    }

    cachedSweepStart = entry->sweepMode ? entry->previousTranslate : entry->translate;

    // Bounding sphere: union of the current and (for sweep mode) start
    // placements, each evaluated as a non-sweep primitive with its matching
    // model so capsule-style getSphere(model) calls see the right translate.
    // (SweepSignedDistanceFunction::getSphere reuses the outer model for both
    // ends, which collapses the trail for capsule-likes — so it is not used
    // here.) Padded for the optional effect displacement like the CPU effect
    // constructors (bias += amplitude * factor).
    glm::vec3 boundsCenter(entry->translate);
    float boundsRadius = glm::length(entry->scale);
    computeBrushBounds(*entry, cachedSweepStart, boundsCenter, boundsRadius);

    sceneRenderer->brushRenderer->updateParams(frameIndex, *entry, cachedSweepStart,
                                               boundsCenter, boundsRadius,
                                               getWidth(), getHeight());

    // Advance the sweep start for the next frame's trail.
    if (entry->sweepMode) entry->previousTranslate = entry->translate;
}

void MyApp::applyBrushEventToLayer(const BrushEntry& entry, const glm::vec3& sweepStart, Layer targetLayer) {
    if (!world || !sceneRenderer || useRemote) return;

    AddSignedDistanceOperation addOp;
    DeleteSignedDistanceOperation deleteOp;
    PaintSignedDistanceOperation paintOp;
    const SignedDistanceOperation &brushOp = [&]() -> const SignedDistanceOperation & {
        switch (entry.brushMode) {
            case 1:  return deleteOp;
            case 2:  return paintOp;
            default: return addOp;
        }
    }();

    ensureLayerHandlers();
    if (!world->activeScene().validLayer(targetLayer)) {
        std::cerr << "[applyBrushEventToLayer] invalid targetLayer " << static_cast<int>(targetLayer) << std::endl;
        return;
    }
    Octree* octreePtr = world->scene().getLayerOctree(targetLayer);
    if (!octreePtr) return;
    Octree& octree = *octreePtr;

    Octree::OctreeNodeDataHandler updateHandler, deleteHandler;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        if (targetLayer < 0 || static_cast<size_t>(targetLayer) >= collectors.size()) return;
        updateHandler = collectors[targetLayer]->updateHandler;
        deleteHandler = collectors[targetLayer]->deleteHandler;
    }

    Transformation model(entry.scale, entry.translate, entry.rot);
    SimpleBrush simpleBrush(entry.materialIndex, entry.hsv);
    std::unique_ptr<SvgBrush> svgBrush;
    if (entry.sdfType == 10) {
        if (std::shared_ptr<const SvgShape> shape = loadSvgShapeCached(entry.svgPath, nullptr)) {
            svgBrush = std::make_unique<SvgBrush>(shape, model, entry.materialIndex, entry.hsv,
                                                  entry.svgTintFromFill);
        }
    }
    const TexturePositionPainter& brush =
        svgBrush ? static_cast<const TexturePositionPainter&>(*svgBrush)
                 : static_cast<const TexturePositionPainter&>(simpleBrush);

    Simplifier simplifier(0.95f, 0.2f, true);
    auto applyEntry = [&](SignedDistanceFunction& wrappedFunc) {
        applyBrushWithEffect(entry, wrappedFunc, octree, brushOp, model, brush, simplifier,
                             updateHandler, deleteHandler);
    };
    forEachBrushSDF(entry, model, sweepStart, entry.minSize, "[applyBrushEventToLayer]", applyEntry);
}

void MyApp::ensureBrushBase(Layer layer, Octree& tree) {
    if (layer < 0) return;
    if (brushHistories.size() <= static_cast<size_t>(layer)) brushHistories.resize(static_cast<size_t>(layer) + 1);
    LayerBrushHistory& history = brushHistories[static_cast<size_t>(layer)];
    if (history.hasBase) return;

    std::ostringstream stream(std::ios::binary);
    {
        std::shared_lock<std::shared_mutex> lock(tree.treeMutex);
        OctreeFile file(&tree, "brush-undo-base");
        file.writeToStream(stream);
    }
    const std::string raw = stream.str();
    if (raw.empty()) return;

    uLongf compressedSize = compressBound(static_cast<uLong>(raw.size()));
    std::vector<uint8_t> compressed(compressedSize);
    if (compress2(compressed.data(), &compressedSize,
                  reinterpret_cast<const Bytef*>(raw.data()), static_cast<uLong>(raw.size()),
                  Z_BEST_SPEED) != Z_OK) {
        std::cerr << "[BrushUndo] failed to compress the base snapshot for layer " << layer << "\n";
        return;
    }
    compressed.resize(compressedSize);
    history.baseSnapshot = std::move(compressed);
    history.baseRawBytes = raw.size();
    history.hasBase = true;
    std::cout << "[BrushUndo] layer " << layer << " base snapshot: " << history.baseSnapshot.size() / 1048576.0
              << " MiB compressed (" << history.baseRawBytes / 1048576.0 << " MiB raw)\n";
}

void MyApp::recordBrushEvent(Layer layer, const BrushHistoryEvent& event) {
    if (layer < 0) return;
    if (brushHistories.size() <= static_cast<size_t>(layer)) brushHistories.resize(static_cast<size_t>(layer) + 1);
    brushHistories[static_cast<size_t>(layer)].events.push_back(event);
}

void MyApp::undoBrushLayer(Layer layer, bool group) {
    if (useRemote) {
        std::cout << "[BrushUndo] ignored in remote mode (server owns the scene)\n";
        return;
    }
    if (!world || !sceneRenderer || layer < 0 || static_cast<size_t>(layer) >= brushHistories.size()) return;
    LayerBrushHistory& history = brushHistories[static_cast<size_t>(layer)];
    if (!history.hasBase || history.events.empty()) {
        std::cout << "[BrushUndo] layer " << layer << " has nothing to undo\n";
        return;
    }
    if (!world->activeScene().validLayer(layer)) return;
    Octree* tree = world->scene().getLayerOctree(layer);
    if (!tree) return;

    size_t undoCount = 1;
    if (group) {
        // Chain consecutive events whose gap is below 1 s, so an entire rapid
        // stroke is undone (not just the last second of it). A pause of 1 s or
        // more starts a new group.
        double newer = history.events.back().timestamp;
        while (undoCount < history.events.size()) {
            const double older = history.events[history.events.size() - 1 - undoCount].timestamp;
            if (newer - older >= 1.0) break;
            ++undoCount;
            newer = older;
        }
    }

    // Quiesce the background tessellation, drop pending uploads destined for
    // this layer (they reference the pre-undo node pointers), upload the other
    // layers' outstanding work, then free this layer's GPU chunks.
    if (sceneProcessThread.joinable()) sceneProcessThread.join();
    {
        std::deque<PendingMeshData> pending;
        sceneRenderer->drainPendingMeshes(pending, std::numeric_limits<size_t>::max());
        for (auto it = pending.begin(); it != pending.end();) {
            if (it->layer == layer) it = pending.erase(it);
            else ++it;
        }
        if (!pending.empty()) {
            sceneRenderer->processPendingMeshes(this, camera.getPosition(), pending);
        }
    }
    sceneRenderer->removeAllLayerMeshes(layer);

    std::vector<uint8_t> raw(history.baseRawBytes);
    uLongf rawSize = static_cast<uLongf>(raw.size());
    const int rc = uncompress(raw.data(), &rawSize, history.baseSnapshot.data(),
                              static_cast<uLong>(history.baseSnapshot.size()));
    if (rc != Z_OK || rawSize != history.baseRawBytes) {
        std::cerr << "[BrushUndo] failed to decompress the base snapshot for layer " << layer
                  << " (zlib " << rc << ")\n";
        return;
    }
    {
        std::unique_lock<std::shared_mutex> lock(tree->treeMutex);
        tree->reset();
        std::istringstream stream(std::string(reinterpret_cast<const char*>(raw.data()), raw.size()),
                                  std::ios::binary);
        OctreeFile file(tree, "brush-undo-base");
        file.readFromStream(stream);
        if (!tree->root) {
            std::cerr << "[BrushUndo] restored base has no root for layer " << layer << "\n";
            return;
        }
    }

    history.events.resize(history.events.size() - undoCount);

    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        if (layer < static_cast<Layer>(collectors.size()) && collectors[layer]) collectors[layer]->clear();
    }

    undoReplaying = true;
    for (const BrushHistoryEvent& event : history.events) {
        applyBrushEventToLayer(event.entry, event.sweepStart, layer);
    }
    undoReplaying = false;

    dispatchAllEvents();
    // Re-emit synchronously: the restored tree has fresh node pointers, so the
    // old tessellation dedup must be dropped and every chunk re-requested.
    {
        Scene& scene = world->activeScene();
        scene.invalidateTessellationCache(layer);
        Octree::OctreeNodeDataHandler updateHandler;
        {
            std::lock_guard<std::mutex> lock(layersSyncMutex_);
            if (layer >= 0 && static_cast<size_t>(layer) < collectors.size() && collectors[layer]) {
                updateHandler = collectors[layer]->updateHandler;
            }
        }
        if (updateHandler) {
            scene.reemitLayer(layer, updateHandler);
            dispatchLayerEvents(layer);
        }
    }
    std::cout << "[BrushUndo] layer " << layer << ": undid " << undoCount << " event(s), replayed "
              << history.events.size() << " (base " << history.baseSnapshot.size() / 1048576.0
              << " MiB compressed)\n";
}

void MyApp::clearBrushHistory() {
    brushHistories.clear();
    undoReplaying = false;
    undoKeyWasDown = false;
}

void MyApp::pollBrushUndo() {
    GLFWwindow* glfwWindow = getWindow();
    if (!glfwWindow) return;
    const bool ctrl = glfwGetKey(glfwWindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(glfwWindow, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    const bool shift = glfwGetKey(glfwWindow, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(glfwWindow, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    const bool z = glfwGetKey(glfwWindow, GLFW_KEY_Z) == GLFW_PRESS;
    const bool combo = ctrl && z;
    if (combo && !undoKeyWasDown) {
        const bool uiCapture = ImGui::GetIO().WantCaptureKeyboard;
        const bool radial = radialMenu && radialMenu->IsVisible();
        if (!uiCapture && !radial) {
            const BrushEntry* entry = brushManager.getSelectedEntry();
            if (entry) undoBrushLayer(static_cast<Layer>(entry->targetLayer), shift);
        }
    }
    undoKeyWasDown = combo;
}

void MyApp::applyBrushToScene() {
    if (!world || !sceneRenderer) return;
    // Brush edits target the LOCAL octrees; in remote mode there is nothing
    // local to edit (the server owns the SDF), so refuse instead of silently
    // building invisible local content.
    if (useRemote) {
        std::cout << "[MyApp::applyBrushToScene] ignored in remote mode (server owns the scene)\n";
        return;
    }

    const BrushEntry* selectedEntry = brushManager.getSelectedEntry();
    if (!selectedEntry) return;
    const BrushEntry entry = *selectedEntry;
    const Layer targetLayer = static_cast<Layer>(entry.targetLayer);
    if (!world->activeScene().validLayer(targetLayer)) {
        std::cerr << "[MyApp::applyBrushToScene] invalid targetLayer " << entry.targetLayer << std::endl;
        return;
    }

    if (!undoReplaying) {
        if (Octree* tree = world->scene().getLayerOctree(targetLayer)) {
            ensureBrushBase(targetLayer, *tree);
            BrushHistoryEvent event;
            event.entry = entry;
            event.sweepStart = cachedSweepStart;
            event.timestamp = glfwGetTime();
            computeBrushBounds(entry, cachedSweepStart, event.boundsCenter, event.boundsRadius);
            // Record only after the apply succeeds? The apply has no return
            // value; recording first preserves ordering even if the SDF is a
            // no-op, which is what the history-based replay expects.
            recordBrushEvent(targetLayer, event);
        }
    }

    applyBrushEventToLayer(entry, cachedSweepStart, targetLayer);
    dispatchAllEvents();

    if (entry.sweepMode) {
        BrushEntry* mutableEntry = brushManager.getSelectedEntry();
        if (mutableEntry) mutableEntry->previousTranslate = mutableEntry->translate;
    }
}

// Ensure pending texture generation requests are flushed after a frame is submitted
// so array-layer transitions happen outside of active draw command buffers.

void MyApp::action() {
    // Remote mode has no local tessellation pass; restart live dispatch.
    if (useRemote) {
        stopRemoteDispatch();
        // Wait for the GPU to finish all in-flight work before clearing GPU resources
        deviceWaitIdle();
        startRemoteDispatchLoop();
        return;
    }
    // Join any previous background tessellation thread
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    // Wait for the GPU to finish all in-flight work before clearing GPU resources
    deviceWaitIdle();

    // Tessellate chunks in a background thread. Solid and water are handled on
    // separate threads so both layers tessellate truly in parallel (water no
    // longer waits for solid to finish).
    startTessellationThreads("[MyApp::action] Scene chunk tessellation complete\n");
}

void MyApp::resetSceneState() {
    clearBrushHistory();
    stopRemoteDispatch();
    if (sceneProcessThread.joinable()) sceneProcessThread.join();
    deviceWaitIdle();
    processPendingCommandBuffers();

    if (sceneRenderer) {
        sceneRenderer->removeAllMeshes();
        world->chunkManager().removeAll();
        if (sceneRenderer->debugCubeRenderer) sceneRenderer->debugCubeRenderer->clearCubes();
        if (sceneRenderer->debugSDFRenderer) sceneRenderer->debugSDFRenderer->clearCubes();
        if (sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->clearAllInstances();
        }
        if (sceneRenderer->sdfRenderer) {
            sceneRenderer->sdfRenderer->clearLava();
        }
    }

    // Remote mode has no local octree content; the GPU/collector clearing
    // above is the whole reset (all callers that rebuild octrees are guarded
    // to local mode). Keep the live stream flowing.
    if (useRemote) {
        startRemoteDispatchLoop();
        return;
    }

    world->scene().resetAllLayers();

    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        for (auto& c : collectors) c->clear();
    }
}

void MyApp::dispatchLayerEvents(Layer layer) {
    std::shared_ptr<UniqueChangeCollector> collector;
    Octree::OctreeNodeDataHandler onAdd, onDel;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= collectors.size()) return;
        if (static_cast<size_t>(layer) >= layerAddHandlers.size()) return;
        collector = collectors[static_cast<size_t>(layer)];
        onAdd = layerAddHandlers[static_cast<size_t>(layer)];
        onDel = layerRemoveHandlers[static_cast<size_t>(layer)];
    }
    if (!collector) return;
    // Disabled layers drop queued events (no tessellation while hidden).
    // repopulateLayer() re-queues on re-enable.
    if (world && !world->activeScene().layerEnabled(layer)) {
        collector->clear();
        return;
    }
    // The scene server is a second consumer of the same deduped changes, so a
    // client sees every edit the local renderer does.
    Octree::OctreeNodeDataHandler serverAdd, serverDel;
    if (sceneServer && sceneServer->running()) {
        serverAdd = [this, layer](const OctreeNodeData& nd) {
            sceneServer->pushUpsert(nd, static_cast<int>(layer));
        };
        serverDel = [this, layer](const OctreeNodeData& nd) {
            sceneServer->pushDelete(nd, static_cast<int>(layer));
        };
    }
    collector->dispatch(
        [&](const OctreeNodeData& nd) {
            if (onAdd) onAdd(nd);
            if (serverAdd) serverAdd(nd);
        },
        [&](const OctreeNodeData& nd) {
            if (onDel) onDel(nd);
            if (serverDel) serverDel(nd);
        });
}

void MyApp::dispatchAllEvents() {
    // Snapshot size under lock; per-layer dispatch re-locks to copy.
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        n = collectors.size();
    }
    for (size_t i = 0; i < n; ++i)
        dispatchLayerEvents(static_cast<Layer>(i));
}

void MyApp::ensureLayerHandlers() {
    if (!world || !sceneRenderer) return;
    std::lock_guard<std::mutex> lock(layersSyncMutex_);
    Scene& scene = world->activeScene();
    const size_t n = std::max<size_t>(scene.layerCount(), 2);
    sceneRenderer->ensureLayerStates(n);
    if (collectors.size() == n && layerAddHandlers.size() == n && layerRemoveHandlers.size() == n) return;
    // Grow (stable addresses via shared_ptr) or shrink (only the tail is ever
    // removed — LayersWidget restricts removal to the last layer, so earlier
    // indices stay valid; pending events for the dropped layer are discarded).
    // Old collectors stay alive via shared_ptr copies held by in-flight
    // dispatches.
    while (collectors.size() > n) collectors.pop_back();
    while (collectors.size() < n)
        collectors.push_back(std::make_shared<UniqueChangeCollector>());
    // Rebuild ALL handlers: build() captures PublishTarget references to the
    // layer's heap-allocated state, so previously built handlers stay valid,
    // but rebuilding is cheap and keeps indices in sync after add/remove.
    layerAddHandlers.assign(n, {});
    layerRemoveHandlers.assign(n, {});
    Scene* sceneForChanges = &world->activeScene();
    const float minSize = 30.0f;
    for (size_t i = 0; i < n; ++i) {
        const Layer layer = static_cast<Layer>(i);
        auto handlers = build(
            sceneRenderer, this, sceneForChanges, layer, minSize,
            &sceneRenderer->layerGenPool(layer),
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->layerChunks(layer),
                sceneRenderer->layerMutex(layer),
                sceneRenderer->layerPendingDeletes(layer)
            });
        layerAddHandlers[i] = handlers.first;
        layerRemoveHandlers[i] = handlers.second;
    }
}

void MyApp::syncLayers() {
    if (!world) return;
    Scene* active = &world->activeScene();
    {
        static size_t knownLayerCount = 0;
        const size_t layerCount = active->layerCount();
        if (layerCount != knownLayerCount) {
            if (knownLayerCount != 0) clearBrushHistory();
            knownLayerCount = layerCount;
        }
    }
    if (layersWidget) {
        layersWidget->setScene(active);
        layersWidget->setAllowStructureEdit(!useRemote);
    }
    if (brush3dWidget) brush3dWidget->setScene(active);
    if (octreeExplorerWidget) octreeExplorerWidget->setScene(active);
    ensureLayerHandlers();
}

void MyApp::repopulateLayer(Layer layer) {
    if (!world || !sceneRenderer) return;
    ensureLayerHandlers();
    Scene& scene = world->activeScene();
    if (!scene.validLayer(layer) || !scene.layerEnabled(layer)) return;
    // The GPU slots were freed; the octree still holds the data but the
    // tessellation dedup cache considers it up-to-date, so invalidate first
    // (otherwise the re-emit below is skipped as a no-op).
    scene.invalidateTessellationCache(layer);
    Octree::OctreeNodeDataHandler updateHandler;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= collectors.size() || !collectors[static_cast<size_t>(layer)])
            return;
        updateHandler = collectors[static_cast<size_t>(layer)]->updateHandler;
    }
    scene.reemitLayer(layer, updateHandler);
    // Dispatch on a worker thread (same pattern as startTessellationThreads):
    // the collector replay enqueues tessellation onto the layer's gen pool
    // and returns quickly; the main thread uploads results progressively.
    // Detached: dispatchLayerEvents only holds shared_ptr copies, so a
    // concurrent add/remove cannot free anything under it.
    std::thread([this, layer]() { dispatchLayerEvents(layer); }).detach();
}

// ── H3 (perf report 29): debug overlay collection gate ─────────────────────
// The two debug renderers' enabled bits are the only gate on the per-chunk
// subtree walks (SceneRenderer::collectDebugForChunk). Applying the settings
// here starts/stops collection; the draw path keeps checking
// settings.showSDFDebug / settings.showBoundingBoxes exactly as before, so
// nothing about what is drawn changes — with both off (the default) the walks
// and the ~478 MB of retained records simply never happen.
void MyApp::applyDebugOverlaySettings() {
    if (!sceneRenderer) return;
    const bool sdfOn = settings.showSDFDebug;
    const bool bboxOn = settings.showBoundingBoxes;
    const bool sdfWas = sceneRenderer->debugSDFRenderer && sceneRenderer->debugSDFRenderer->isEnabled();
    const bool bboxWas = sceneRenderer->boundingBoxRenderer && sceneRenderer->boundingBoxRenderer->isEnabled();
    // Steady state: four relaxed atomic loads, nothing else.
    if (sdfOn == sdfWas && bboxOn == bboxWas) return;

    // Off clears the retained per-chunk records inside setEnabled(false).
    if (sceneRenderer->debugSDFRenderer) sceneRenderer->debugSDFRenderer->setEnabled(sdfOn);
    if (sceneRenderer->boundingBoxRenderer) sceneRenderer->boundingBoxRenderer->setEnabled(bboxOn);

    // On: repopulate every live chunk once. Without this, an overlay switched
    // on mid-session would stay empty until each chunk happened to rebuild.
    if ((sdfOn && !sdfWas) || (bboxOn && !bboxWas)) recollectDebugOverlays();
}

// One-shot re-collection for every live chunk. reemitLayer() walks the octree
// and hands us each chunk node (the same nodes loadScene/repopulateLayer see)
// without re-tessellating or uploading anything; collectDebugForChunk() then
// runs the enabled debug walks for the chunks that currently own a published
// mesh — exactly the set processNodeLayer collected for, so the overlay matches
// a build-time collection. Runs on the main thread, which is also where every
// octree mutation happens (brush apply / scene load), so no node can be freed
// while the walks run. Cost: the two walks over the live chunks, once, on the
// toggle that enables an overlay.
void MyApp::recollectDebugOverlays() {
    if (!world || !sceneRenderer) return;
    Scene& scene = world->activeScene();
    const size_t layerCount = scene.layerCount();
    for (size_t i = 0; i < layerCount; ++i) {
        const Layer layer = static_cast<Layer>(i);
        if (!scene.layerEnabled(layer)) continue;

        // Snapshot the live chunk ids (published chunks and their ancestor
        // cells) under the layer's map lock.
        std::unordered_set<NodeID> live;
        {
            std::lock_guard<std::recursive_mutex> lock(sceneRenderer->layerMutex(layer));
            const auto& chunks = sceneRenderer->layerChunks(layer);
            live.reserve(chunks.size());
            for (const auto& kv : chunks) live.insert(kv.first);
        }
        if (live.empty()) continue;

        // reemitLayer invokes the handler under the octree's shared lock, so
        // only copy the descriptors here; the debug walks run afterwards and
        // take that lock themselves.
        std::vector<OctreeNodeData> nodes;
        scene.reemitLayer(layer, [&nodes](const OctreeNodeData& nd) { nodes.push_back(nd); });
        for (OctreeNodeData& nd : nodes) {
            if (!nd.node) continue;
            const NodeID nid = reinterpret_cast<NodeID>(nd.node);
            if (live.find(nid) == live.end()) continue;
            sceneRenderer->collectDebugForChunk(scene, layer, nid, nd, &sceneRenderer->layerGenPool(layer));
        }
    }
}

// Builds (once) the GPU droplet simulation + renderer for the active scene.
//
// The collision table is the cropped octree SDF region: the same node layout
// the GPU topology pass consumes (entry 0 = world root, corner samples with
// INFINITY sentinels, children indices), so a droplet collides with the
// identical trilinear SDF the terrain mesh is generated from. Rendering and
// physics are independent: this table is never rebuilt because the render LoD
// or the droplet's render settings change, only when the scene changes.
void MyApp::ensureDroplets() {
    if (useRemote || sceneRenderer == nullptr || !world) return;
    if (!dropletRenderer) dropletRenderer = std::make_unique<DropletRenderer>();
    if (!dropletRenderer->ready()) dropletRenderer->init(this);
    if (!particleSim) particleSim = std::make_shared<ParticlePhysicsSim>();
    if (particleSim->ready()) return;

    // Region: a vertical column around the spawn XZ wide enough for the spawn
    // ball plus its spread, spanning the whole octree height so the terrain
    // surface is inside the table at every spawn position. Droplets that leave
    // it are retired by the finalize pass, so it never has to cover the map.
    const float lateral = std::max(16.0f, settings.dropletSpawnRadius) * 2.5f;
    const glm::vec3 rootMin = world->scene().lodRootMin(0);
    const glm::vec3 rootMax = rootMin + glm::vec3(world->scene().lodRootSize(0));
    dropletRegionMin = glm::vec3(-lateral, rootMin.y, -lateral);
    dropletRegionMax = glm::vec3(lateral, rootMax.y, lateral);

    std::vector<OctreeGpuNode> nodes;
    if (!world->scene().buildSdfRegionTable(0, dropletRegionMin, dropletRegionMax, nodes,
                                            ParticlePhysicsSim::kMaxRegionNodes) ||
        nodes.empty()) {
        std::cerr << "[MyApp] droplet simulation disabled: no SDF region table for layer 0\n";
        return;
    }
    particleSim->init(this);
    particleSim->uploadRegion(nodes);
    particleSim->setParticleShape(settings.dropletRadius, 1.0f);
    // Spawn on top of the height map: the table is uploaded, so the terrain
    // height at the spawn XZ can be queried on the host.
    const float terrainY = particleSim->terrainHeightAt(0.0f, 0.0f);
    dropletSpawnCenter_ = glm::vec3(0.0f, terrainY + std::max(5.0f, settings.dropletSpawnHeight), 0.0f);
    particleSim->requestReset();
    dropletShapeRadius = settings.dropletRadius;
    std::cout << "[MyApp] droplet simulation ready: " << nodes.size() << " SDF nodes, "
              << settings.dropletCount << " particles, terrain Y=" << terrainY
              << ", spawn centre Y=" << dropletSpawnCenter_.y << "\n";
}

void MyApp::generateMap() {
    if (useRemote) {
        std::cout << "[MyApp::generateMap] ignored in remote mode (rendering server scene)\n";
        return;
    }
    resetSceneState();
    ensureLayerHandlers();

    // Build the octree (CPU only, no tessellation)
    MainSceneLoader loader;
    loader.setVideoBrushIndex(videoCubeMaterialIndex);
    loader.setSdfGivenBrushIndex(sdfGivenMaterialIndex);
    std::vector<Octree::OctreeNodeDataHandler> ups, dels;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        ups.reserve(collectors.size());
        dels.reserve(collectors.size());
        for (auto& c : collectors) { ups.push_back(c->updateHandler); dels.push_back(c->deleteHandler); }
    }
    world->scene().loadScene(loader, ups, dels);
    std::cout << "[MyApp::generateMap] Octree construction complete\n";

    // Tessellate chunks in background threads, one per layer.
    startTessellationThreads("[MyApp::generateMap] Scene chunk tessellation complete\n");
    ensureDroplets();
}

void MyApp::loadRemoteScene() {
    if (!useRemote || !remoteScene) return;
    ensureLayerHandlers();
    // The loader callback is meaningless remotely (no local octree to build);
    // RemoteScene ignores it and replays the server snapshot instead.
    MainSceneLoader loader;
    std::vector<Octree::OctreeNodeDataHandler> ups, dels;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        ups.reserve(collectors.size());
        dels.reserve(collectors.size());
        for (auto& c : collectors) { ups.push_back(c->updateHandler); dels.push_back(c->deleteHandler); }
    }
    remoteScene->loadScene(loader, ups, dels);
    if (remoteScene->failed() || !remoteScene->isConnected()) {
        // Fall back to LocalScene: drop partial remote events, disconnect,
        // and generate locally instead. The renderer never saw a mix because
        // nothing was dispatched yet (dispatch starts below, only on success).
        std::cout << "[MyApp::loadRemoteScene] snapshot failed; falling back to LocalScene\n";
        {
            std::lock_guard<std::mutex> lock(layersSyncMutex_);
            for (auto& c : collectors) c->clear();
        }
        remoteScene->disconnect();
        remoteScene.reset();
        useRemote = false;
        if (world) world->setActiveScene(nullptr);
        generateMapPending = true;
        return;
    }
    std::cout << "[MyApp::loadRemoteScene] snapshot replayed; starting live dispatch\n";
    startRemoteDispatchLoop();
}

void MyApp::loadSceneFromFile(const std::string& path) {
    if (useRemote) {
        std::cout << "[MyApp::loadSceneFromFile] ignored in remote mode (rendering server scene)\n";
        return;
    }
    resetSceneState();
    ensureLayerHandlers();

    std::vector<Octree::OctreeNodeDataHandler> ups, dels;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        ups.reserve(collectors.size());
        dels.reserve(collectors.size());
        for (auto& c : collectors) { ups.push_back(c->updateHandler); dels.push_back(c->deleteHandler); }
    }
    world->scene().load(path, ups, dels, &settings);
    std::cout << "[MyApp::loadSceneFromFile] Octree loaded from '" << path << "'\n";

    // Every layer tessellates on its own thread.
    startTessellationThreads("[MyApp::loadSceneFromFile] Scene tessellation complete\n");

    // New scene: the old droplet region table belongs to the previous octree.
    // Drop it so the lazy bring-up rebuilds the region from the loaded scene.
    if (particleSim) {
        particleSim->destroy(this);
        particleSim.reset();
    }
    ensureDroplets();
}
void MyApp::postSubmit() {
    // H3 (perf report 29): follow the overlay settings every frame. The widget
    // writes settings.showSDFDebug / showBoundingBoxes directly, so this is the
    // single place that turns them into collection gates (a toggle-on also runs
    // the one-shot re-collection). Steady state is four relaxed atomic loads.
    applyDebugOverlaySettings();
    pollBlenderExport();
    // Droplet diagnostics: throttled counter readback (never in the hot path).
    if (settings.dropletsEnabled && settings.dropletStats && particleSim && particleSim->ready()) {
        dropletStatsClock += std::max(lastFrameDelta, 0.0f);
        if (dropletStatsClock >= 1.0) {
            dropletStatsClock = 0.0;
            const ParticlePhysicsSim::Stats ds = particleSim->readStats();
            const ParticlePhysicsSim::Probe probe = particleSim->probeParticles();
            std::cout << "[droplets] pairContacts=" << ds.pairContacts
                      << " terrain+escaped=" << ds.terrainAndEscaped
                      << " gridOverflow=" << ds.gridOverflow
                      << " maxContactsPerParticle=" << ds.maxContactsPerParticle
                      << " | active=" << probe.activeCount
                      << " y[min=" << probe.minY << " mean=" << probe.meanY
                      << " max=" << probe.maxY << "]";
            if (probe.probed > 0u) {
                const glm::vec4& p0 = probe.positions[0];
                const glm::vec4& v0 = probe.velocities[0];
                std::cout << " | p0=(" << p0.x << "," << p0.y << "," << p0.z << ")"
                          << " r=" << p0.w << " |v0|=" << v0.w;
            }
            std::cout << std::endl;
        }
    }
    // Lazy bring-up: toggling the master gate on after startup builds the sim.
    if (settings.dropletsEnabled && sceneRenderer && (!particleSim || !particleSim->ready())) {
        ensureDroplets();
    }

    // Scene server: start/stop with the settings toggle (or VULKAN_SERVE=1).
    // Only a LocalScene can be served; a remote scene already has a server.
    if (settings.serveScene && !sceneServer && world) {
        if (auto* local = dynamic_cast<LocalScene*>(&world->activeScene())) {
            sceneServer = std::make_unique<SceneServer>(
                *local, static_cast<uint16_t>(std::clamp(settings.serveScenePort, 1024, 65535)));
            try {
                sceneServer->start();
            } catch (const std::exception& e) {
                std::cerr << "[MyApp] scene server failed to start: " << e.what() << std::endl;
                sceneServer.reset();
                settings.serveScene = false;
            }
        } else {
            std::cerr << "[MyApp] scene server unavailable: the active scene is remote" << std::endl;
            settings.serveScene = false;
        }
    } else if (!settings.serveScene && sceneServer) {
        sceneServer->stop();
        sceneServer.reset();
    }
    // VULKAN_STREAM_DIAG=1: periodic chunk-streaming counters (loaded/tracked
    // per layer + the cull's visible counts) so a headless run can be checked
    // for permanently missing regions without reading the UI overlay.
    if (std::getenv("VULKAN_STREAM_DIAG") != nullptr && sceneRenderer) {
        static auto lastDiag = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        static int diagTick = 0;
        if (std::chrono::duration<double>(now - lastDiag).count() >= 20.0) {
            lastDiag = now;
            const size_t solidLoaded = sceneRenderer->solidRenderer
                ? sceneRenderer->solidRenderer->getIndirectRenderer().getMeshCount() : 0;
            const size_t waterLoaded = sceneRenderer->waterRenderer
                ? sceneRenderer->waterRenderer->getIndirectRenderer().getMeshCount() : 0;
            const StreamStats ss = sceneRenderer->streamStats();
            std::cout << "[stream-diag] tick=" << ++diagTick
                      << " solidLoaded=" << solidLoaded << " waterLoaded=" << waterLoaded
                      << " tracked=" << sceneRenderer->getRegisteredModelCount()
                      << " opaqueVisible=" << sceneRenderer->getLastOpaqueVisible()
                      << " transparentVisible=" << sceneRenderer->getLastTransparentVisible()
                      << " | topoReady=" << ss.topologyReady
                      << " topoUpToDate=" << ss.topologyUpToDate
                      << " topoUnavailable=" << ss.topologyUnavailable
                      << " built=" << ss.residentBuilt
                      << " empty=" << ss.residentEmpty
                      << " failed=" << ss.residentFailed
                      << " cpuFallback=" << ss.cpuFallback
                      << std::endl;
        }
    }
    if (textureMixer) {
        textureMixer->flushPendingRequests(this);
        textureMixer->pollPendingGenerations(this);
    }

    // Video textures: advance playback clocks, publish decoded frames into the
    // array layers and log diagnostics on request. This never blocks: when the
    // requested frame is not decoded yet the layer keeps the previous frame.
    if (videoTextureService) {
        videoTextureService->update(std::clamp(lastFrameDelta, 0.0f, 0.25f));
    }

    if (brushApplyToScenePending) {
        brushApplyToScenePending = false;
        applyBrushToScene();
    }

    if (generateMapPending) {
        generateMapPending = false;
        generateMap();
        // Pool utilization telemetry (perf report 22 C1): one line per pool
        // plus <25% warnings while non-empty.
        if (sceneRenderer) sceneRenderer->logMemoryUtilization();
    }

    if (loadScenePending) {
        loadScenePending = false;
        loadSceneFromFile(pendingLoadPath);
        if (sceneRenderer) sceneRenderer->logMemoryUtilization();
    }

    if (remoteLoadPending) {
        remoteLoadPending = false;
        loadRemoteScene();
        if (sceneRenderer) sceneRenderer->logMemoryUtilization();
    }
}
