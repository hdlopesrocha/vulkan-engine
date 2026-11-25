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
#include <future>
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
#include "vulkan/includes/debug/DebugModes.hpp"
#include "vulkan/streaming/PublishTarget.hpp"
#include "vulkan/renderer/SceneRenderer.hpp"
#include "vulkan/renderer/SceneDescriptorLayout.hpp"
#include "vulkan/renderer/SceneQueues.hpp"
#include "vulkan/renderer/RendererUtils.hpp"
#include "utils/LocalScene.hpp"
#include "utils/OctreeRaymarchTable.hpp"
#include "utils/GrassPlacement.hpp"
#include "utils/LandBrush.hpp"
#include "sdf/core/SDF.hpp"
#include "math/geometry/BoundingCube.hpp"
#include "brushPainters/SvgBrush.hpp"
#include "sdf/primitives/SvgPolygonDistanceFunction.hpp"
#include "sdf/primitives/SvgShape.hpp"
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
#include "widgets/WindParticlesWidget.hpp"
#include "widgets/RaymarchWidget.hpp"
#include "widgets/GrassWidget.hpp"
#include "sdf/types/SdfScene.hpp"
#include "widgets/OctreeExplorerWidget.hpp"
#include "widgets/Brush3dWidget.hpp"
#include "widgets/BrushInterpolationWidget.hpp"
#include "widgets/LayersWidget.hpp"
#include "widgets/MusicWidget.hpp"
#include "widgets/components/FilePicker.hpp"
#include "sdf/operations/AddSignedDistanceOperation.hpp"
#include "sdf/operations/DeleteSignedDistanceOperation.hpp"
#include "sdf/operations/PaintSignedDistanceOperation.hpp"
#include "sdf/operations/SweepSignedDistanceFunction.hpp"
#include "sdf/operations/InterpolatedDistanceFunction.hpp"
#include "utils/MainSceneLoader.hpp"
#include "space/UniqueChangeCollector.hpp"
#include "space/OctreeFile.hpp"
#include "types/Settings.hpp"
#include "types/WaterSettings.hpp"
#include "utils/GraphicsSettingsCommand.hpp"
#include "math/WindFieldCpu.hpp"
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
              Layer::Id layer, ThreadPool* genPool, const PublishTarget& target) {

    // Solid/Water terrain is consumed directly by the SDF raymarch table: the
    // octree shape only has to materialize its SDF nodes. No chunk topology
    // pass runs, no vertex/index span is reserved and no VBO/IBO is written
    // for these layers. Video layers keep their dedicated textured renderer.
    Layer* layerObj = scene ? scene->findLayer(layer) : nullptr;
    const bool raymarchOnlyLayer =
        layerObj && layerObj->renderer() != LayerRendererType::Video;

    Octree::OctreeNodeDataHandler onAdded = [renderer, app, scene, layer, genPool, target, raymarchOnlyLayer](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);

        if (raymarchOnlyLayer) {
            // Terrain never touches the chunk mesh pools. processNodeLayer()
            // takes its raymarch-only branch: debug overlays plus the
            // disposable CPU surface queued for vegetation placement.
            OctreeNodeData nodeCopy = nd;
            renderer->processNodeLayer(
                *scene, layer, nid, nodeCopy,
                [](Layer::Id, NodeID, const LoDMesh&) {},
                genPool);
            return;
        }

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
            [renderer, cid, nodeCopy, target](Layer::Id layer_, NodeID nid_, const LoDMesh& lodMesh) {
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
            genPool);
    };

    Octree::OctreeNodeDataHandler onDeleted = [renderer, app, scene, target, raymarchOnlyLayer](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);

        if (raymarchOnlyLayer) {
            // No chunk slot/dedup record exists for an SDF-only terrain node;
            // just drop any retained debug-overlay records for its address.
            if (renderer->debugCubeRenderer) renderer->debugCubeRenderer->removeCubeForNode(nid);
            if (renderer->debugSDFRenderer) renderer->debugSDFRenderer->removeCubesForNode(nid);
            return;
        }

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
    // Best-effort: a failed spawn must not affect startup.
    (void)std::system(cmd.c_str());
}

class MyApp : public VulkanApp, public IEventHandler {
public:
    // ── S1 staged bring-up ────────────────────────────────────────────────
    // setupStep() advances exactly one entry per frame. Long steps start
    // asynchronous work (texture decode/upload, the four renderer groups) and
    // return; the matching Poll step is retried until the futures are ready.
    enum class BootStage : uint32_t {
        Env = 0,
        SceneInfrastructure,
        WorldAndBrushes,
        VegetationRenderer,
        VegetationTextures,
        TexturesBegin,
        TexturesPoll,
        TexturesFinish,
        RenderersBegin,
        RenderersPoll,
        RenderersFinish,
        TextureViewer,
        UiWidgets,
        RegisterWidgets,
        InputAndCamera,
        BrushAndLayerWidgets,
        ProfilingResources,
        ScheduleLoad,
        Done
    };
    BootStage bootStage = BootStage::Env;
    bool setupDiag_ = false;
    std::chrono::steady_clock::time_point setupDiagLast_{};
    std::chrono::steady_clock::time_point bootStart_{};
    // Async texture load started by setupTexturesBegin(). The future returns
    // the number of loaded layers; it is polled, never waited on, by the boot
    // loop so the window keeps presenting frames while 5*N maps decode/upload.
    std::future<size_t> textureLoadFuture_;
    std::vector<TextureTriple> textureTriplesForLoad_;
    uint32_t textureLayerCount_ = 0;
    // Set by abortSetup() so a Quit during the texture stage stops starting new
    // decode batches/uploads; the worker returns the layers completed so far.
    std::atomic<bool> textureLoadCancel_{false};

    // ── S2: background content build ─────────────────────────────────────
    // The octree build/load runs on its own CPU-only thread. The main thread
    // keeps rendering; postSubmit() polls contentBuildDone_ and finalizes
    // (start tessellation, droplets, settings) after present.
    std::thread contentThread;
    std::atomic<bool> contentLoading_{false};
    std::atomic<bool> contentBuildDone_{false};
    std::atomic<bool> contentBuildFailed_{false};
    std::atomic<bool> contentReady_{false};
    std::atomic<bool> contentStreaming_{false};
    std::atomic<bool> sceneDispatchRunning_{false};
    std::atomic<uint32_t> contentChunksExpected_{0};
    std::chrono::steady_clock::time_point contentStart_{};
    bool contentIsFile_ = false;
    std::string contentPath_;
    std::string contentError_;
    Settings contentSettingsCopy_;
    bool contentHasSettings_ = false;
    Settings settings;
    // Last grass mode applied to the renderers; the mode selector only
    // pushes on change so the Grass/Impostors widget toggles are not
    // overridden every frame.
    Settings::GrassMode appliedGrassMode = Settings::GrassMode::None;
    SceneRenderer * sceneRenderer = nullptr;
    // Octree SDF raymarch table (the ONLY terrain geometry path). Rebuilt on
    // a debounce when the layered scene changes and uploaded into the current
    // frame slot on revision change.
    OctreeRaymarchTable raymarchTable_;
    uint64_t raymarchTableVersion_ = 0;
    // Accumulated cloud wind displacement (world metres): integrated on the
    // host from the shared wind field's ambient fBm velocity so the cloud
    // pattern travels with the wind (SkyUniform::cloudWindDrift).
    glm::vec2 cloudWindDrift_{0.0f};
    uint64_t raymarchLayerSignature_ = 0;
    bool raymarchSignaturePrimed_ = false;
    bool raymarchDirty_ = false;
    float raymarchDirtyTime_ = 0.0f;
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
    std::shared_ptr<BrushInterpolationWidget> brushInterpolationWidget;
    bool brushInterpolationWasEnabled_ = false;
    // Shared brush entries edited by Brush3dWidget (owned by MyApp)
    Brush3dManager brushManager;
    // Cached sweep start position so applyBrushToScene uses the same pair as the preview
    glm::vec3 cachedSweepStart = glm::vec3(0.0f);
    // Same, for the interpolation target's independent sweep trail.
    glm::vec3 cachedTargetSweepStart_ = glm::vec3(0.0f);

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
    float profileSdf = 0.0f;        // SDF engine ray-march pass (slots 22-23)
    float profileOctreeMarch = 0.0f; // octree SDF raymarch draw (slots 0-1)
    // SDF march counters (opt-in, set=1 binding 10): fragment atomics gated by
    // SdfRenderer::setProfileCounters; read back with a 3-frame slot latency.
    bool sdfProfilingEnabled_ = false;
    float profileSdfCpu = 0.0f; // CPU record time of the async SDF task
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
    // Wind-trail particle visualization editor (WindRenderer): trail duration/
    // color/width + the shared wind parameters.
    std::shared_ptr<WindParticlesWidget> windParticlesWidget;
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
    // S3: one topology service per scene layer. Each lane owns its input
    // buffers, descriptor set and submission slot; getGlobal() binds each
    // layer worker to one lane, removing the old process-wide mutex convoy.
    std::vector<std::shared_ptr<OctreeTopologyCompute>> octreeTopologyLanes;
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
    // Phase 0 deterministic benchmark path (VULKAN_CAMERA_PATH). The pose is
    // derived from the start pose and a frame counter, so two runs with the
    // same frame count see exactly the same cameras regardless of input/fps.
    enum class BenchCameraPath { None, Static, Pan5, Pan60, Dolly, Strafe };
    BenchCameraPath benchCameraPath_ = BenchCameraPath::None;
    glm::vec3 benchCameraBasePos_{0.0f};
    glm::quat benchCameraBaseOri_{1.0f, 0.0f, 0.0f, 0.0f};
    uint64_t benchCameraFrame_ = 0;
    // Optional headless A/B capture: close after N ready frames so the exit
    // screenshot is taken at the same post-load frame in each run.
    uint64_t benchReadyFrames_ = 0;
    uint64_t benchExitAfterFrames_ = 0;
    void applyBenchCameraPath();
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
    // Live layer ids the handlers were built for. Ids are stable and never
    // reused, so a remove+add with the same count must still rebuild.
    std::vector<Layer::Id> knownLayerIds_;
    mutable std::mutex layersSyncMutex_;
    ThreadPool asyncThreadPool{1}; // single worker for per-frame back-face pass

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

    // S1: staged texture bring-up. Begin allocates the arrays and starts the
    // decode/upload worker; Poll is non-blocking; Finish does the main-thread
    // ImGui/material wiring that needs the loaded layer count.
    void setupTexturesBegin() {

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
        textureTriplesForLoad_.clear();
        textureTriplesForLoad_.reserve(textureTriples_.size());
        for (const SceneTextureTriple& t : textureTriples_) {
            textureTriplesForLoad_.push_back({t.albedo.c_str(), t.normal.c_str(), t.height.c_str(),
                                              t.rough.c_str(), t.ao.c_str()});
        }

        uint32_t layerCount = static_cast<uint32_t>(textureTriplesForLoad_.size()) + kSceneMixerCount + kScenePaintMargin;
        layerCount = std::max(layerCount, kMinLayers);
        std::cout << "[MyApp::setupTextures] texture arrays: " << layerCount
                  << " layers at " << arraySize << "^2 ("
                  << textureTriplesForLoad_.size() << " triples + " << kSceneMixerCount
                  << " mixers + " << kScenePaintMargin << " paint margin)\n";
        textureArrayManager.allocate(layerCount, arraySize, arraySize, this);

        // S1: decode + upload on a worker. loadTriples() already parallelizes
        // the 5*N decodes and submits one batch; the boot loop polls the
        // future instead of blocking the main thread for the whole upload.
        loadedTextureLayers = 0;
        textureLoadCancel_.store(false, std::memory_order_relaxed);
        textureLoadFuture_ = std::async(std::launch::async, [this]() {
            return textureArrayManager.loadTriples(this, textureTriplesForLoad_,
                                                   &textureLoadCancel_);
        });
        // Ensure mixer descriptor sets are updated with newly loaded arrays
        
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, &textureArrayManager);
        // Surface Cartesian vertices are extracted only on the GPU (the
        // topology pass writes the packed pools). There is no CPU vertex
        // solver and no readback path to compare against any more.
        // S3: three independent topology lanes (opaque/water/video). Init is
        // cheap now that the pipeline cache is warm; each lane still owns all
        // its GPU state, so workers never share a mutex or descriptor set.
        constexpr uint32_t kTopologyLanes = 3;
        octreeTopologyLanes.clear();
        octreeTopologyLanes.reserve(kTopologyLanes);
        for (uint32_t lane = 0; lane < kTopologyLanes; ++lane) {
            auto service = std::make_shared<OctreeTopologyCompute>();
            service->init(this);
            octreeTopologyLanes.push_back(std::move(service));
        }
        textureMixer->attachTextureArrayManager(&textureArrayManager);
        // Record into member so UI can display counts
        textureLayerCount_ = layerCount;
    }

    bool setupTexturesPoll() {
        if (!textureLoadFuture_.valid()) return true;
        if (textureLoadFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return false;
        try {
            loadedTextureLayers = textureLoadFuture_.get();
        } catch (const std::exception& e) {
            std::cerr << "[MyApp::setupTextures] asynchronous load failed: " << e.what() << "\n";
            loadedTextureLayers = 0;
        } catch (...) {
            std::cerr << "[MyApp::setupTextures] asynchronous load failed (unknown)\n";
            loadedTextureLayers = 0;
        }
        return true;
    }

    void setupTexturesFinish() {
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
        uint32_t editableLayer = (loadedTextureLayers < textureLayerCount_) ? loadedTextureLayers : 0u;

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

        // Headless true-displacement bring-up: VULKAN_DISPLACEMENT=1 applies
        // the new material path to every mapping-enabled material with one
        // amplitude/frequency/strength set. Used by the displacement test /
        // benchmark script; the interactive path edits the same fields in the
        // Textures widget.
        if (const char* env = std::getenv("VULKAN_DISPLACEMENT")) {
            if (env[0] != '0') {
                auto envFloat = [](const char* name, float fallback) {
                    if (const char* v = std::getenv(name)) return static_cast<float>(std::atof(v));
                    return fallback;
                };
                const float amp = envFloat("VULKAN_DISPLACEMENT_AMP", 4.0f);
                const float freq = envFloat("VULKAN_DISPLACEMENT_FREQ", 1.0f);
                const float strength = envFloat("VULKAN_DISPLACEMENT_STRENGTH", 1.0f);
                const float offset = envFloat("VULKAN_DISPLACEMENT_OFFSET", 0.0f);
                const float bounds = envFloat("VULKAN_DISPLACEMENT_BOUNDS", 0.0f);
                const char* texEnv = std::getenv("VULKAN_DISPLACEMENT_TEXTURE");
                const int dispTexture = texEnv ? std::atoi(texEnv) : -1;
                size_t applied = 0;
                for (MaterialProperties& m : materials) {
                    if (!m.mappingMode) continue;
                    m.displacementAmplitude = amp;
                    m.displacementFrequency = freq;
                    m.displacementOffset = offset;
                    m.displacementStrength = strength;
                    m.displacementBounds = bounds;
                    m.displacementTexture = dispTexture;
                    ++applied;
                }
                std::cout << "[MyApp] true displacement: " << applied
                          << " mapping materials, amp=" << amp
                          << " freq=" << freq << " strength=" << strength
                          << " offset=" << offset << " bounds=" << bounds
                          << " texture=" << dispTexture << std::endl;
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

        // SceneRenderer::init() (called right after setupTextures) writes the
        // complete static/per-frame set-0 bindings with the now-allocated
        // materials and brush-map buffers, and registers the texture-allocation
        // listener for later array reallocations. Calling
        // updateTextureDescriptorSet() here would run before the shadow maps,
        // material handles and water buffers are wired, so it could only log
        // skipped bindings — and init would overwrite the set anyway.

        // Perf report 23 C1: committed vs. initialized texture memory. The
        // scene renderer logs it again after every scene load/generation.
        textureArrayManager.logMemoryUtilization("main");
    }

    // Compatibility entry point: run the whole staged bring-up synchronously.
    // VulkanApp::run() does not call this; it pumps setupStep() between frames.
    void setup() override {
        while (!setupStep()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // S1: one boot stage per call. Returning false means "poll me again next
    // frame" (the stage started async work and is waiting for it); true means
    // the whole setup is complete.
    bool setupStep() override {
        if (bootStage == BootStage::Env) {
            // VULKAN_SETUP_DIAG=1: wall time per stage. The first stamp is the
            // start of the boot, so the first line includes applySetupEnvOverrides.
            setupDiag_ = std::getenv("VULKAN_SETUP_DIAG") != nullptr;
            setupDiagLast_ = std::chrono::steady_clock::now();
            bootStart_ = setupDiagLast_;
            applySetupEnvOverrides();
            setupDiagStep("environment overrides");
            advanceBoot();
            return false;
        }

        switch (bootStage) {
        case BootStage::SceneInfrastructure:
            createSceneInfrastructure([this](const char* name) { setupDiagStep(name); });
            advanceBoot();
            break;

        case BootStage::WorldAndBrushes:
            createWorldAndBrushes();
            advanceBoot();
            break;

        case BootStage::VegetationRenderer:
            // Must run before setupVegetationTextures: the impostor service
            // shares the wind UBO + descriptor layout created here.
            if (sceneRenderer && sceneRenderer->vegetationRenderer)
                sceneRenderer->vegetationRenderer->init(this);
            setupDiagStep("vegetation renderer init");
            advanceBoot();
            break;

        case BootStage::VegetationTextures:
            setupVegetationTextures();
            setupDiagStep("vegetation textures");
            advanceBoot();
            break;

        case BootStage::TexturesBegin:
            setupTexturesBegin();
            setupDiagStep("texture decode/upload started");
            advanceBoot();
            break;

        case BootStage::TexturesPoll:
            if (!setupTexturesPoll()) return false; // keep presenting frames
            setupDiagStep("texture decode/upload (wall)");
            advanceBoot();
            break;

        case BootStage::TexturesFinish:
            setupTexturesFinish();
            playSetupAlert("setup-textures");
            setupDiagStep("textures + materials");
            advanceBoot();
            break;

        case BootStage::RenderersBegin:
            // Offscreen target scales must be known before the async groups
            // create those targets.
            if (sceneRenderer) {
                sceneRenderer->setVegetationRenderScale(settings.vegetationRenderScale);
                sceneRenderer->beginInit(this, &textureArrayManager);
            }
            setupDiagStep("renderer init groups started");
            advanceBoot();
            break;

        case BootStage::RenderersPoll:
            if (sceneRenderer && !sceneRenderer->pollInit()) return false;
            setupDiagStep("renderer init groups (wall)");
            advanceBoot();
            break;

        case BootStage::RenderersFinish:
            finishRendererInit();
            playSetupAlert("setup-renderers");
            setupDiagStep("renderer post-init (video bind, impostor rewire)");
            advanceBoot();
            break;

        case BootStage::TextureViewer:
            createTextureViewerPanel();
            setupDiagStep("texture viewer");
            advanceBoot();
            break;

        case BootStage::UiWidgets:
            createUiWidgets();
            setupDiagStep("UI widgets");
            advanceBoot();
            break;

        case BootStage::RegisterWidgets:
            registerUiWidgets();
            // All panels the app UI expects now exist; drawFrame() switches
            // from the minimal boot panel to renderImGui() while the last
            // (short) stages finish.
            uiReady = true;
            setupDiagStep("UI registration");
            advanceBoot();
            break;

        case BootStage::InputAndCamera:
            setupInputAndCamera();
            setupDiagStep("input + camera");
            advanceBoot();
            break;

        case BootStage::BrushAndLayerWidgets:
            createBrushAndLayerWidgets();
            setupDiagStep("brush + layer widgets");
            advanceBoot();
            break;

        case BootStage::ProfilingResources:
            createProfilingResources();
            setupDiagStep("profiling resources");
            advanceBoot();
            break;

        case BootStage::ScheduleLoad:
            scheduleInitialSceneLoad();
            playSetupAlert("setup-ready");
            setupDiagStep("schedule initial scene load");
            bootStage = BootStage::Done;
            return true;

        case BootStage::Done:
            return true;

        case BootStage::Env:
            break; // handled above
        }
        return false;
    }

    const char* setupStepName() const override {
        switch (bootStage) {
        case BootStage::Env:                 return "Applying environment overrides";
        case BootStage::SceneInfrastructure: return "Creating scene infrastructure";
        case BootStage::WorldAndBrushes:     return "Creating world and layer handlers";
        case BootStage::VegetationRenderer:  return "Initializing vegetation renderer";
        case BootStage::VegetationTextures:  return "Preparing vegetation textures";
        case BootStage::TexturesBegin:       return "Starting texture decode";
        case BootStage::TexturesPoll:        return "Decoding and uploading scene textures";
        case BootStage::TexturesFinish:      return "Finalizing textures and materials";
        case BootStage::RenderersBegin:      return "Starting renderer pipelines";
        case BootStage::RenderersPoll:       return "Creating renderer pipelines";
        case BootStage::RenderersFinish:     return "Finalizing renderers and pools";
        case BootStage::TextureViewer:       return "Creating texture viewer";
        case BootStage::UiWidgets:           return "Creating UI widgets";
        case BootStage::RegisterWidgets:     return "Registering UI widgets";
        case BootStage::InputAndCamera:      return "Setting up input and camera";
        case BootStage::BrushAndLayerWidgets:return "Creating brush and layer widgets";
        case BootStage::ProfilingResources:  return "Creating profiling resources";
        case BootStage::ScheduleLoad:        return "Scheduling scene content";
        case BootStage::Done:                return "Ready";
        }
        return "Starting up";
    }

    // Called from drawFrame() once per boot frame (events are polled by the
    // run() loop). Gives the user a way out even before input is initialized.
    void updateBoot(float /*deltaTime*/) override {
        if (!getWindow()) return;
        if (glfwGetKey(getWindow(), GLFW_KEY_ESCAPE) == GLFW_PRESS) requestClose();
    }

    // Minimal ImGui panel shown before the full UI exists.
    void renderBootUi() override {
        drawBootPanel(false);
    }

    // S1: if the window is closed (or a stage throws) while an async boot task
    // is in flight, join it before cleanup() destroys the resources it uses.
    void abortSetup() override {
        if (textureLoadFuture_.valid()) {
            textureLoadCancel_.store(true, std::memory_order_relaxed);
            try { (void)textureLoadFuture_.get(); } catch (const std::exception& e) {
                std::cerr << "[MyApp::abortSetup] texture load failed: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "[MyApp::abortSetup] texture load failed (unknown)\n";
            }
        }
        if (sceneRenderer && sceneRenderer->initStarted()) {
            while (!sceneRenderer->pollInit())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // Futures are ready, so no renderer worker is still using device
            // resources. A partially finished init is safe to tear down: every
            // created object was registered with the resource manager.
        }
    }

    // ── setup() sections ─────────────────────────────────────────────────────
    void advanceBoot() {
        bootStage = static_cast<BootStage>(static_cast<uint32_t>(bootStage) + 1u);
    }

    void setupDiagStep(const char* name) {
        if (!setupDiag_) return;
        const auto now = std::chrono::steady_clock::now();
        std::cout << "[setup-diag] " << name << ": "
                  << std::chrono::duration<double, std::milli>(now - setupDiagLast_).count()
                  << " ms" << std::endl;
        setupDiagLast_ = now;
    }

    void drawBootPanel(bool compact) {
        const ImGuiIO& io = ImGui::GetIO();
        const double elapsed = bootStart_.time_since_epoch().count() != 0
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - bootStart_).count()
            : 0.0;
        const float progress = std::min(0.99f,
            static_cast<float>(static_cast<uint32_t>(bootStage)) /
            static_cast<float>(static_cast<uint32_t>(BootStage::Done)));

        ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoSavedSettings;
        if (compact) {
            ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, io.DisplaySize.y - 12.0f),
                                    ImGuiCond_Always, ImVec2(1.0f, 1.0f));
            flags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
            ImGui::SetNextWindowBgAlpha(0.75f);
            ImGui::Begin("##boot_compact", nullptr, flags);
            ImGui::Text("Starting up: %s", setupStepName());
            ImGui::ProgressBar(progress, ImVec2(220.0f, 0.0f));
        } else {
            ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                                    ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(560.0f, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Starting up", nullptr, flags | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::TextUnformatted("Vulkan Engine");
            ImGui::Separator();
            ImGui::Text("Stage: %s", setupStepName());
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f));
            ImGui::Text("Elapsed: %.1f s", elapsed);
            ImGui::Spacing();
            ImGui::TextDisabled("The world loads after the renderers are ready.");
            ImGui::Spacing();
            if (ImGui::Button("Quit")) requestClose();
        }
        ImGui::End();
    }

    // S2: non-modal content progress. Drawn while the octree is built on
    // contentThread and while its chunks stream in. The camera and the rest of
    // the UI remain fully interactive.
    void drawContentPanel() {
        const ImGuiIO& io = ImGui::GetIO();
        const bool building = contentLoading_.load(std::memory_order_acquire);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_AlwaysAutoResize |
                                 ImGuiWindowFlags_NoFocusOnAppearing |
                                 ImGuiWindowFlags_NoNav |
                                 ImGuiWindowFlags_NoSavedSettings;
        ImGui::SetNextWindowBgAlpha(0.80f);
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, io.DisplaySize.y - 12.0f),
                                ImGuiCond_Always, ImVec2(1.0f, 1.0f));
        ImGui::Begin("##content_progress", nullptr, flags);

        if (building) {
            const double elapsed = contentStart_.time_since_epoch().count() != 0
                ? std::chrono::duration<double>(std::chrono::steady_clock::now() - contentStart_).count()
                : 0.0;
            ImGui::TextUnformatted(contentIsFile_ ? "Loading scene..." : "Building world...");
            if (contentIsFile_) {
                ImGui::TextDisabled("%.72s", contentPath_.c_str());
            }
            // Negative fraction: ImGui draws its indeterminate animation.
            ImGui::ProgressBar(-1.0f, ImVec2(260.0f, 0.0f));
            ImGui::Text("Elapsed: %.1f s", elapsed);
            ImGui::TextDisabled("Camera and UI stay interactive.");
        } else {
            const size_t published = sceneRenderer ? sceneRenderer->getStreamedMeshCount() : 0;
            const uint32_t expected = contentChunksExpected_.load(std::memory_order_relaxed);
            ImGui::Text("Streaming chunks: %zu", published);
            if (expected > 0) {
                const float fraction = std::min(1.0f,
                    static_cast<float>(published) / static_cast<float>(expected));
                ImGui::ProgressBar(fraction, ImVec2(260.0f, 0.0f));
                ImGui::TextDisabled("~%u chunk events queued", expected);
            } else {
                ImGui::ProgressBar(-1.0f, ImVec2(260.0f, 0.0f));
            }
        }
        ImGui::End();
    }

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

        // VULKAN_WIND_RAYMARCH=1 enables the wind-velocity isosurface raymarch
        // (headless validation of WindRaymarchRenderer; the default stays off).
        if (const char* env = std::getenv("VULKAN_WIND_RAYMARCH")) {
            if (env[0] != '0') settings.windRaymarchEnabled = true;
        }

        // VULKAN_CLOUDS=1 starts with volumetric clouds on (headless validation
        // of the cloud raymarch and its wind drift; the default stays off).
        if (const char* env = std::getenv("VULKAN_CLOUDS")) {
            if (env[0] != '0') settings.cloudsEnabled = true;
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

        // Displacement test scene: a deterministic close view of the slab and
        // sphere unless the caller supplied VULKAN_CAMERA.
        if (std::getenv("VULKAN_DISPLACEMENT_TEST") != nullptr) {
            // Clean deterministic capture: no particles/droplets/vegetation
            // layered over the displaced slab and sphere.
            settings.windParticlesEnabled = false;
            settings.dropletsEnabled = false;
            settings.vegetationEnabled = false;
            settings.cloudsEnabled = false;
            if (std::getenv("VULKAN_CAMERA") == nullptr) {
                // Math::eulerToQuat(0,0,0) looks along -Z; -90 deg yaw turns
                // that toward +X, where the slab/sphere sit from this camera.
                camera = Camera(glm::vec3(-800.0f, 320.0f, 0.0f),
                                Math::eulerToQuat(-90.0f, -15.0f, 0.0f));
                std::cout << "[MyApp] displacement test camera: (-800, 320, 0)" << std::endl;
            }
        }
        // Headless A/B switches for layer isolation (diagnostics only): each
        // forces the matching Settings toggle off/on before the first frame.
        if (std::getenv("VULKAN_NO_SOLID") != nullptr) settings.renderSolid = false;
        if (std::getenv("VULKAN_NO_VEG") != nullptr) settings.vegetationEnabled = false;
        if (std::getenv("VULKAN_NO_SHADOWS") != nullptr) settings.enableShadows = false;
        // True-displacement A/B switches for headless runs.
        if (const char* env = std::getenv("VULKAN_DISPLACEMENT")) {
            if (env[0] == '0') settings.displacementEnabled = false;
            else settings.displacementEnabled = true;
        }
        if (const char* env = std::getenv("VULKAN_DISPLACEMENT_AO"))
            settings.displacementAOEnabled = env[0] != '0';
        if (const char* env = std::getenv("VULKAN_DISPLACEMENT_CASTER"))
            settings.displacementShadowCasterEnabled = env[0] != '0';
        if (std::getenv("VULKAN_NO_TERRAIN_SHADOWS") != nullptr)
            settings.displacementShadowCasterEnabled = false;
        // VULKAN_OCTREE_SCREEN_EPS=<px>: A/B capture of the
        // screen-space hit tolerance. 0 keeps the world-space epsilon.
        if (const char* env = std::getenv("VULKAN_OCTREE_SCREEN_EPS")) {
            settings.octreeRaymarchScreenEps =
                std::clamp(static_cast<float>(std::atof(env)), 0.0f, 8.0f);
        }
        // VULKAN_BENCH_EXIT_AFTER=<frames>: request a clean exit (and the
        // normal exit screenshot) after N frames rendered once content is
        // ready. Used by scripts/compare_raymarch_eps.sh; 0 disables.
        if (const char* env = std::getenv("VULKAN_BENCH_EXIT_AFTER")) {
            const long frames = std::atol(env);
            benchExitAfterFrames_ = frames > 0 ? static_cast<uint64_t>(frames) : 0;
        }
        // VULKAN_CAMERA_PATH=static|pan5|pan60|dolly|strafe: deterministic
        // benchmark camera derived from VULKAN_CAMERA (or the default pose).
        // The path advances once per update; dolly/strafe use 1 m and 0.5 m
        // per frame at the nominal 60 fps benchmark cadence.
        if (const char* path = std::getenv("VULKAN_CAMERA_PATH")) {
            const std::string s(path);
            if (s == "static") benchCameraPath_ = BenchCameraPath::Static;
            else if (s == "pan5") benchCameraPath_ = BenchCameraPath::Pan5;
            else if (s == "pan60") benchCameraPath_ = BenchCameraPath::Pan60;
            else if (s == "dolly") benchCameraPath_ = BenchCameraPath::Dolly;
            else if (s == "strafe") benchCameraPath_ = BenchCameraPath::Strafe;
            else if (!s.empty())
                std::cerr << "[Camera] unknown VULKAN_CAMERA_PATH '" << s
                          << "'; expected static|pan5|pan60|dolly|strafe" << std::endl;
            if (benchCameraPath_ != BenchCameraPath::None) {
                benchCameraBasePos_ = camera.getPosition();
                benchCameraBaseOri_ = camera.getOrientation();
                benchCameraFrame_ = 0;
                std::cout << "[Camera] deterministic path '" << s << "' from ("
                          << benchCameraBasePos_.x << ", " << benchCameraBasePos_.y
                          << ", " << benchCameraBasePos_.z << ")" << std::endl;
            }
        }

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

    // S1: final main-thread wiring after the four async renderer groups have
    // completed. Runs between frames (the boot loop calls it after present),
    // so ImGui descriptor creation in ShadowRenderer::init is safe.
    void finishRendererInit() {
        if (!sceneRenderer) return;
        sceneRenderer->finishInit(this, &textureArrayManager, &materialManager, waterParams);

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
            sceneRenderer->shadowMapper.get());
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
        windWidget = std::make_shared<WindWidget>(sceneRenderer->vegetationRenderer.get(), &settings);
        windParticlesWidget = std::make_shared<WindParticlesWidget>(
            &settings, sceneRenderer->vegetationRenderer.get());
        raymarchWidget = std::make_shared<RaymarchWidget>(
            sceneRenderer->sdfRenderer.get(), &camera,
            sceneRenderer->vegetationRenderer.get(), &settings);
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
        widgetManager.addWidget(windParticlesWidget);
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
        shadowParams.update(light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);
        
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

        // Separate target-SDF window for two-SDF interpolation.
        brushInterpolationWidget = std::make_shared<BrushInterpolationWidget>(brushManager);
        widgetManager.addWidget(brushInterpolationWidget);

        // Dynamic scene layers + per-layer renderer selection.
        layersWidget = std::make_shared<LayersWidget>(&world->activeScene(), sceneRenderer,
            [this]() { syncLayers(); },
            [this](Layer::Id layer) { repopulateLayer(layer); });
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
    void applyBrushEventToLayer(const BrushEntry& entry, const glm::vec3& sweepStart, Layer::Id layer);
    // Serialize/restore the layer base and manage the per-layer brush stack.
    void ensureBrushBase(Layer::Id layer, Octree& tree);
    void recordBrushEvent(Layer::Id layer, const BrushHistoryEvent& event);
    void undoBrushLayer(Layer::Id layer, bool group);
    void clearBrushHistory();
    void pollBrushUndo();
    // S2: CPU-only background build. startLocalContentLoad() resets the scene,
    // snapshots the collectors and launches the build thread; the render loop
    // polls contentBuildDone_ and calls finalizeContentLoad() after present.
    void startLocalContentLoad(bool fromFile, std::string path);
    void finalizeContentLoad();
    // Start the scene server if settings.serveScene is set and it is not
    // running. Called before tessellation dispatch so the server sees every
    // chunk event of a fresh content build.
    void startSceneServerIfEnabled();
    // Builds the GPU droplet sim + renderer for the active scene (once).
    void ensureDroplets();
    void action();
    // Shared scene state reset (join thread, wait GPU, clear meshes/octrees/handlers)
    void resetSceneState();
    // Replay the deduplicated scene change events into the renderer handlers
    // (replaces UniqueOctreeChangeHandler::handleEvents).
    void dispatchLayerEvents(Layer::Id layer);
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
    void repopulateLayer(Layer::Id layer);

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

    // ── Octree SDF raymarch (only terrain path) ───────────────────────────
    // CPU signature of the solid-layer list (ids/enabled/renderer/octree
    // pointer) so add/remove/replace/re-renderer changes mark the node table
    // dirty without per-frame tree walks.
    uint64_t raymarchLayerSignature() const;
    // Debounced rebuild + publish of the node table. Mutations mark
    // raymarchDirty_ where they happen (brush apply, scene load), never from
    // chunk-upload traffic: tessellation does not change octree SDF data.
    void refreshRaymarchTable(float deltaTime);
    // Places grass tufts directly from the raymarch table surface nodes (the
    // terrain has no Cartesian mesh). Runs on the main thread after a table
    // rebuild and replaces the previous vegetation set.
    void generateVegetationFromRaymarchTable();

    // Start the background tessellation thread for the current scene. Every
    // layer dispatches on its own thread so all layers tessellate truly in
    // parallel. `completionLog` is printed verbatim once all threads join.
    void startTessellationThreads(const char* completionLog) {
        ensureLayerHandlers();
        std::vector<Layer::Id> ids;
        if (world) {
            Scene& scene = world->activeScene();
            for (size_t i = 0; i < scene.layerCount(); ++i)
                if (const Layer* l = scene.layerAt(i)) ids.push_back(l->id());
        }
        sceneDispatchRunning_.store(true, std::memory_order_relaxed);
        sceneProcessThread = std::thread([this, completionLog, ids]() {
            std::vector<std::thread> workers;
            workers.reserve(ids.size());
            for (Layer::Id id : ids)
                workers.emplace_back([this, id]() { dispatchLayerEvents(id); });
            for (auto& t : workers) t.join();
            sceneDispatchRunning_.store(false, std::memory_order_release);
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
                                            (world && !useRemote && !contentLoading_.load() &&
                                             world->scene().findLayer(LAYER_OPAQUE))
                                                ? world->scene().findLayer(LAYER_OPAQUE)->octree() : nullptr);
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

        // Phase 0: deterministic benchmark camera overrides any input motion
        // after the event queue has been processed.
        applyBenchCameraPath();

        // Phase 3 capture helper: count only post-load frames, then request
        // the normal clean exit so the existing exit-screenshot path saves the
        // final swapchain image.
        if (benchExitAfterFrames_ > 0 && contentReady_.load(std::memory_order_acquire) &&
            !contentLoading_.load(std::memory_order_acquire)) {
            if (++benchReadyFrames_ >= benchExitAfterFrames_) {
                std::cout << "[Bench] capture frame reached (" << benchExitAfterFrames_
                          << " ready frames); requesting exit" << std::endl;
                benchExitAfterFrames_ = 0;
                requestClose();
            }
        }

        shadowParams.update(light, camera.getViewProjectionMatrix(), settings.nearPlane, settings.farPlane);

        // Drain the pending mesh queue populated by the background scene-loading
        // thread.  GPU uploads happen here on the main thread so newly generated
        // chunks become visible progressively without blocking the render loop.
        // Process pending meshes at a controlled rate (10 per frame).
        // Chunks closest to the camera are uploaded first. Drains pending
        // entries from the ONE shared queue.
        // S2: while the content worker owns the octrees, skip every scene
        // reader (syncLayers, maxChunkLod, drain/publish). There are no new
        // chunks during the build, and the tree must not be read concurrently.
        if (sceneRenderer && !isLoading && !contentLoading_.load(std::memory_order_acquire)) {
            // Dynamic layers: pick up LayersWidget add/remove without restart.
            syncLayers();
            // Video-layer mesh uploads stage their indirect commands into the
            // renderer's per-frame cull slot, so the slot must already match the
            // frame being prepared (solid/water set theirs in preRenderPass; the
            // video pass only culls later, after the uploads).
            if (sceneRenderer->videoRenderer) {
                sceneRenderer->videoRenderer->getIndirectRenderer().setCullFrame(getCurrentFrame());
            }
            std::deque<PendingMeshData> pendingBatch;
            sceneRenderer->drainPendingMeshes(pendingBatch, 16);
            sceneRenderer->processPendingMeshes(this, pendingBatch);
        }

        // Octree SDF raymarch: rebuild the node table on a debounce and keep
        // every frame slot's revision current. Uploading here is safe: the
        // frame-slot fence for getCurrentFrame() was waited before update().
        // S2: the table reads the octree; build it only once content is final.
        if (sceneRenderer && sceneRenderer->octreeRaymarchRenderer &&
            contentReady_.load(std::memory_order_acquire) &&
            !contentLoading_.load(std::memory_order_acquire)) {
            refreshRaymarchTable(deltaTime);
            sceneRenderer->octreeRaymarchRenderer->upload(getCurrentFrame(), raymarchTable_);
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
            sceneRenderer->vegetationRenderer->processPendingChunks(64);
        }

        mainTime += deltaTime;
        lastFrameDelta = deltaTime;
        if (sceneRenderer && sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->setWindTime(mainTime);
            sceneRenderer->vegetationRenderer->setImpostorDistance(settings.impostorDistance);
        }
        if (sceneRenderer && sceneRenderer->skyRenderer) {
            sceneRenderer->skyRenderer->setCloudTime(mainTime);
            // Cloud drift: integrate the shared wind field's ambient fBm
            // velocity at a sky reference point so the cloud pattern travels
            // with the SAME wind the vegetation/water/fire/trails sample. The
            // GPU samples the field per pixel; the bulk offset is a path
            // integral, so the host accumulates it with a CPU mirror of the
            // field (math/WindFieldCpu.hpp). Reference = the camera's XZ at a
            // mid-cloud altitude: a local, smooth sample; the accumulated
            // offset keeps the drift continuous as the camera moves.
            if (sceneRenderer->vegetationRenderer) {
                const auto& ws = sceneRenderer->vegetationRenderer->getWindSettings();
                WindFieldCpu::FbmParams fbm;
                fbm.frequency = ws.fbmFrequency;
                fbm.lacunarity = ws.fbmLacunarity;
                fbm.gain = ws.fbmGain;
                fbm.amplitude = ws.fbmAmplitude;
                fbm.vertical = ws.fbmVertical;
                fbm.seed = ws.fbmSeed;
                fbm.speed = ws.speed;
                fbm.octaves = ws.fbmOctaves;
                const glm::vec3 refPoint(camera.getPosition().x, 3000.0f,
                                         camera.getPosition().z);
                const glm::vec3 windVec =
                    WindFieldCpu::windAmbient(refPoint, mainTime, fbm);
                cloudWindDrift_ += glm::vec2(windVec.x, windVec.z) *
                                   std::max(deltaTime, 0.0f);
                sceneRenderer->skyRenderer->setCloudWindDrift(cloudWindDrift_);
            }
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

    void preRenderPass(VkCommandBuffer &commandBuffer) override {

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
                // Group D: indices 22-23 (SDF engine ray-march pass, own CB slots)
                struct { uint64_t value; uint64_t availability; } tsD[2] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 22, 2,
                        sizeof(tsD), tsD, sizeof(tsD[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsD[0].availability) profileSdf = msDiff(tsD[1].value, tsD[0].value);
                }
                // Dedicated octree raymarch interval (slots 0/1).
                struct { uint64_t value; uint64_t availability; } tsO[2] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 0, 2,
                        sizeof(tsO), tsO, sizeof(tsO[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsO[0].availability) profileOctreeMarch = msDiff(tsO[1].value, tsO[0].value);
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
                        profileSdf + profilePostProcess + profileImGui;
                    if (gpuTotal > 40.0f) {
                        std::cout << "[gpu] total=" << gpuTotal
                                  << " depth=" << profileDepthPrepass
                                  << " sky=" << profileSky
                                  << " solid=" << profileSolidDraw
                                  << " veg=" << profileVegetationImpostor
                                  << " water=" << profileWater
                                  << " sdf=" << profileSdf
                                  << " octree=" << profileOctreeMarch
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
        // Screen-space octree raymarch hit tolerance (0 = off, see Settings).
        uboStatic.raymarchScreenEps = settings.octreeRaymarchScreenEps;
        // The new path is only globally active when at least one uploaded
        // material actually configures it. This keeps the default renderer
        // (no material displacement) bit-identical to the old path: the root
        // rejection box stays the stored surface AABB and no extra AO /
        // shadow-caster work is scheduled.
        bool anyMaterialDisplacement = false;
        float maxMaterialDisplacementBound = 0.0f;
        for (const MaterialProperties& m : materials) {
            if (m.displacementStrength > 0.0f && m.displacementAmplitude != 0.0f) {
                anyMaterialDisplacement = true;
                float nmax = std::max(std::abs(-0.5f + m.displacementOffset),
                                      std::abs(0.5f + m.displacementOffset));
                float configured = m.displacementBounds > 0.0f
                    ? m.displacementBounds
                    : std::abs(m.displacementAmplitude) * m.displacementStrength;
                maxMaterialDisplacementBound = std::max(
                    maxMaterialDisplacementBound,
                    std::min(configured,
                             std::abs(m.displacementAmplitude) * m.displacementStrength * nmax));
            } else if (settings.tessellationEnabled && m.mappingMode) {
                maxMaterialDisplacementBound = std::max(
                    maxMaterialDisplacementBound, 0.5f * std::abs(m.tessHeightScale));
            }
        }
        uboStatic.displacementEnabled =
            (settings.displacementEnabled && anyMaterialDisplacement) ? 1u : 0u;
        uboStatic.displacementAOEnabled =
            (settings.displacementAOEnabled && anyMaterialDisplacement) ? 1u : 0u;
        uboStatic.displacementShadowCaster =
            (settings.displacementShadowCasterEnabled && anyMaterialDisplacement) ? 1u : 0u;
        uboStatic.displacementMaxBound = maxMaterialDisplacementBound;
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

        // Reset command buffer state tracker and wire it to all sub-renderers.
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

        // ── Wind-trail particles (WindRenderer) ─────────────────────────────
        // Compute cannot be recorded inside a dynamic-rendering scope, so the
        // particle step runs HERE on the main command buffer (before the
        // swapchain scope is opened). The trail draw happens in draw() after
        // the composite. The simulation reads the SHARED wind field UBO (set=0
        // binding 27, owned by VegetationRenderer), so these particles follow
        // the exact same gust/tornado field as the vegetation, fire and clouds.
        if (sceneRenderer && sceneRenderer->windRenderer) {
            const VkBuffer windFieldBuffer = sceneRenderer->vegetationRenderer
                ? sceneRenderer->vegetationRenderer->getWindFieldBuffer().buffer
                : VK_NULL_HANDLE;
            const bool windEnabled = sceneRenderer->vegetationRenderer
                ? sceneRenderer->vegetationRenderer->getWindSettings().enabled
                : true;
            // The particle far plane is bounded by the camera far plane
            // (Settings > Camera). Clamp the authored values every frame so a
            // camera-far change immediately pulls the band in, even when the
            // Wind Trails widget is closed.
            WindParticlesSettings& windParticles = settings.windParticles;
            const float farPlaneLimit = std::max(settings.farPlane, 1.0f);
            windParticles.spawnFarDistance =
                std::clamp(windParticles.spawnFarDistance, 0.0f, farPlaneLimit);
            windParticles.spawnNearDistance =
                std::clamp(windParticles.spawnNearDistance, 0.0f, windParticles.spawnFarDistance);
            sceneRenderer->windRenderer->dispatch(
                this, commandBuffer, frameIdx, settings.windParticlesEnabled,
                settings.windParticles, uboStatic.viewProjection, uboStatic.invViewProjection,
                uboStatic.viewPosition, settings.farPlane,
                glm::vec2(static_cast<float>(getWidth()), static_cast<float>(getHeight())),
                mainTime, std::clamp(lastFrameDelta, 0.0f, 0.1f), windFieldBuffer, windEnabled);
        }

        // Wind raymarch overlay: params UBO + host->fragment barrier must be
        // recorded OUTSIDE the rendering scope (a pipeline barrier inside
        // dynamic rendering may only use framebuffer-space stages). The draw
        // itself happens in draw().
        if (sceneRenderer && sceneRenderer->windRaymarchRenderer) {
            const VkBuffer windFieldBuffer = sceneRenderer->vegetationRenderer
                ? sceneRenderer->vegetationRenderer->getWindFieldBuffer().buffer
                : VK_NULL_HANDLE;
            sceneRenderer->windRaymarchRenderer->prepare(
                this, commandBuffer, frameIdx, settings.windRaymarchEnabled,
                settings.windRaymarch, uboStatic.invViewProjection, uboStatic.viewPosition,
                sceneRenderer->solidRenderer->getDepthView(frameIdx), windFieldBuffer, mainTime);
        }

        // Octree SDF raymarch path: the ONLY terrain renderer. Every Solid +
        // Water layer is marched from the octree SDFs; there is no raster
        // branch, no toggle and no fallback. The table may be absent on the
        // first frames of a load (debounced rebuild), in which case the
        // fullscreen draw is a no-op and the depth target stays cleared.
        const bool octreeRaymarchActive = sceneRenderer
            && settings.renderSolid
            && sceneRenderer->octreeRaymarchRenderer
            && sceneRenderer->brushRenderer;

        // The raster water pass is deleted. Water is shaded by the same
        // octree march (kind = Water root); only the target clearing remains
        // so the composite never samples stale bytes.
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


        // Launch asynchronous recording+submit for independent offscreen passes
        // using a persistent thread pool to avoid per-frame std::thread creation overhead.
        // Dependency graph (timeline semaphores, one per producer):
        //   cull task      : brush early pass + all GPU culls -> tlCull
        //   shadow task    : shadow map + cascade cull + blur, restores visibleLods/UBO
        //                    (runs after cull on the worker) -> tlShadow
        //                    (not enqueued when shadows are disabled; consumers
        //                    then wait tlCull@v directly)
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
        static uint64_t tlFrameValue = 0;
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
            tlInit = true;
        }
        const uint64_t v = ++tlFrameValue;
        std::future<void> asyncCullFuture;
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
                    this->sceneRenderer->vegetationRenderer->prepareCull();
                if (settings.showSDFDebug && this->sceneRenderer && this->sceneRenderer->debugSDFRenderer)
                    this->sceneRenderer->debugSDFRenderer->registerToIndirect();
                if (settings.showBoundingBoxes && this->sceneRenderer && this->sceneRenderer->boundingBoxRenderer)
                    this->sceneRenderer->boundingBoxRenderer->registerBoundingBoxesToIndirect();
                else if (this->sceneRenderer && this->sceneRenderer->boundingBoxRenderer)
                    this->sceneRenderer->boundingBoxRenderer->clearBoundingBoxesToIndirect();
                this->sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(cullCmd, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
                if (settings.showSDFDebug && this->sceneRenderer && this->sceneRenderer->debugSDFRenderer)
                    this->sceneRenderer->debugSDFRenderer->prepareCull(cullCmd);
                // Signal the single cull timeline semaphore; consumers wait on tlCull@v.
                // Cull is the root producer. The composite does not wait tlCull
                // directly (it is transitively implied through tlSolid and the
                // brush preview's own registered signal). Every consumer that
                // needs cull results waits it explicitly or transitively.
                app->submitCommandBufferAsyncToQueue(cullCmd, app->getGraphicsQueue(), &tlCull, {}, false, {}, {}, v, {}, false);
            });
            asyncCullFuture.get();
            // Restore the per-frame state for the main CB's continued recording.
            this->sceneRenderer->setCmdState(&this->sceneRenderer->frameCmdState);

            // Brush preview is submitted after the shadow task is joined (see
            // below): its camera UBO is restored at the end of that task.
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
            // Water is raymarched into the solid color/depth; there is no
            // raster water-depth image to bind. Point binding 9 at the 1x1
            // water dummy and keep it disabled. Leaving the SdfRenderer's
            // self-depth placeholder there makes the same image both the
            // depth attachment and a sampled binding (VUID 00344).
            if (sceneRenderer->waterRenderer) {
                sceneRenderer->sdfRenderer->setWaterDepth(
                    sceneRenderer->waterRenderer->getDummyWaterColorView(),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);
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

        // --- Brush preview offscreen on its own queue (signals tlBrush) ---
        // Submitted after the shadow task join: with shadows on, the shadow
        // pass temporarily overwrites the main camera UBO and restores it at
        // the end of that task. Waiting on tlShadow (implied by the join +
        // explicit wait below) makes the preview raymarch read the current
        // frame's restored camera instead of the shadow-pass matrices, which
        // made the preview visibly lag behind camera/brush motion.
        // SDF-raymarched brush preview (color + front depth) into the brush
        // targets the composite overlays. No tessellated brush geometry and
        // no dependency on the solid pass: the solid shader evaluates the
        // same brush params (set=1) directly for PAINT/REMOVE. The signal is
        // registered for the composite, which samples the brush targets.
        if (sceneRenderer && sceneRenderer->brushRenderer) {
            updateBrushPreview(frameIdx);
            // Apply the pending stroke HERE, immediately after the preview
            // captured its exact transform/sweep pair. Deferring it to
            // postSubmit (after this frame was recorded) made the painted
            // geometry one full frame behind the brush preview. Rebuild
            // the raymarch table now so the solid raymarch recorded below
            // already contains the stroke.
            if (brushApplyToScenePending && !contentLoading_.load(std::memory_order_acquire)) {
                brushApplyToScenePending = false;
                applyBrushToScene();
            }
            // A brush edit may also arrive from the UI during
            // renderImGui() (undo/redo). The update()-time table refresh
            // has already run, so publish the dirtied table again here:
            // the solid raymarch recorded below then contains this frame's
            // stroke/undo instead of the previous frame's SDF.
            if (raymarchDirty_ && contentReady_.load(std::memory_order_acquire) &&
                !contentLoading_.load(std::memory_order_acquire) &&
                sceneRenderer && sceneRenderer->octreeRaymarchRenderer) {
                refreshRaymarchTable(0.0f);
                sceneRenderer->octreeRaymarchRenderer->upload(
                    getCurrentFrame(), raymarchTable_);
            }
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
                // tlShadow implies tlCull and guarantees the restored camera
                // UBO; shadows off means no tlShadow signal, so wait tlCull.
                VkSemaphore brushWait = settings.enableShadows ? tlShadow : tlCull;
                submitCommandBufferAsyncToQueue(brushCmd, getBrushQueue(), &tlBrush, {brushWait}, true, {}, {v}, v, {}, true);
            }
        }

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
                asyncSkyFuture = asyncThreadPool.enqueue([this, frameIdx, v, sigCopy]() {
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
                            getMainDescriptorSet(), this->sceneRenderer->mainUniformBuffers[frameIdx], this->uboStatic);
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
            asyncSolidFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, v, octreeRaymarchActive]() {
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
                if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
                    // Slots 0/1: dedicated octree raymarch draw interval.
                    vkCmdResetQueryPool(solidCmd, queryPools[frameIdx], 0, 2);
                    vkCmdResetQueryPool(solidCmd, queryPools[frameIdx], 6, 6);
                }

                // ── Sky offscreen moved to its own queue (asyncSkyFuture) so the
                //    equirectangular sky image is produced in parallel with the solid
                //    and shadow passes. The fullscreen sky (render()) still draws into
                //    this solid color pass below. ──

                VkClearValue colorClear{};
                colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                VkClearValue depthClear{};
                depthClear.depthStencil = {1.0f, 0};
                const uint32_t solidW = this->sceneRenderer->solidRenderer->getRenderWidth();
                const uint32_t solidH = this->sceneRenderer->solidRenderer->getRenderHeight();

                // The octree SDF raymarch owns the entire terrain path and
                // writes its own depth (CLEAR + gl_FragDepth). There is no
                // depth prepass and no raster solid draw.

                // Transition solid depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
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

                // ── Terrain color pass: SDF raymarch only ──
                {
                    const VkImageView solidColorView =
                        this->sceneRenderer->solidRenderer->getColorView(frameIdx);
                    const VkImageView solidDepthView =
                        this->sceneRenderer->solidRenderer->getDepthView(frameIdx);
                    // CLEAR + STORE: the raymarch writes color and gl_FragDepth
                    // and later passes sample the depth target.
                    // Publish this frame's host-side table upload and run the
                    // node payload derive pass when the slot got a new
                    // revision. Must happen before vkCmdBeginRendering: HOST
                    // is not a framebuffer-space stage and the compute
                    // dispatch is illegal inside a dynamic rendering scope.
                    this->sceneRenderer->octreeRaymarchRenderer->recordDerive(solidCmd, frameIdx);

                    RendererUtils::beginColorDepthPass(solidCmd, solidColorView, solidDepthView,
                                                       solidW, solidH, colorClear, depthClear);

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 8);
                    if (this->sceneRenderer->skyRenderer) {
                        SkySettings::Mode skyMode = this->sceneRenderer->getSkySettings().mode;
                        this->sceneRenderer->skyRenderer->render(this, solidCmd, getMainDescriptorSet(), skyMode);
                    }
                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 9);

                    if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                        vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 10);

                    if (settings.renderSolid) {
                        // Set=1 brush SDF params: PAINT/REMOVE evaluates the
                        // brush field at each fragment directly.
                        VkDescriptorSet brushParamsSet = this->sceneRenderer->brushRenderer
                            ? this->sceneRenderer->brushRenderer->getParamsDescriptorSet(frameIdx)
                            : VK_NULL_HANDLE;
                        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                            vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                queryPools[frameIdx], 0);
                        this->sceneRenderer->octreeRaymarchRenderer->draw(
                            solidCmd, this, frameIdx, solidW, solidH,
                            getMainDescriptorSet(), brushParamsSet);
                        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                            vkCmdWriteTimestamp(solidCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                                queryPools[frameIdx], 1);

                        // Droplets: direct SDF ray marching in the same pass
                        // after the opaque geometry (depth test against the
                        // solid depth, no depth write, premultiplied coverage
                        // blend). Reads the state the dispatches above wrote.
                        if (settings.dropletsEnabled && dropletRenderer && particleSim &&
                            particleSim->ready() && dropletDrawCount > 0u) {
                            VkImageView skyView = this->sceneRenderer->skyRenderer
                                ? this->sceneRenderer->skyRenderer->getSkyView(frameIdx)
                                : VK_NULL_HANDLE;
                            dropletRenderer->draw(solidCmd, frameIdx, getMainDescriptorSet(),
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
                // waits on it).
                // Shadows disabled: nothing signals tlShadow, so the solid pass
                // waits tlCull@v directly instead (cull buffers + restored UBO).
                // The brush preview is an independent offscreen pass; its signal
                // (tlBrush) goes straight to the composite, not through the
                // solid pass. tlSolid is registered for the composite.
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
        // --- Solid360 360° capture: REMOVED ---
        // The old per-frame 6-face cubemap rasterization (6 culls + 6 raster
        // CBs + join + mip blits) no longer runs; Solid360Renderer and its
        // targets/sync/descriptors were deleted. Water waits tlSolid directly.

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
                // frames never reach this lambda.
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
                        RendererUtils::beginDepthOnlyPass(vegCmd, vegDepthView, w, h, 1.0f);
                        sceneRenderer->vegetationRenderer->drawDepth(this, vegCmd, vegCamPos);
                        RendererUtils::endDepthOnlyPass(vegCmd);
                        // Instance 2: color + depth (depth loaded from prepass)
                        VkClearValue vegColorClear{};
                        vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                        RendererUtils::beginColorDepthPassLoadDepth(vegCmd, vegColorView, vegDepthView,
                                                                    w, h, vegColorClear);
                        sceneRenderer->vegetationRenderer->drawColor(this, vegCmd, vegCamPos,
                                                                     /*drawBillboards=*/true,
                                                                     /*drawImpostors=*/drawImpostors);
                        vkCmdEndRendering(vegCmd);
                    } else {
                        // Grass Raycast impostor-only: one color+depth CLEAR
                        // instance. No billboard depth prepass is needed — the
                        // impostor pipeline writes the captured depth itself
                        // (gl_FragDepth from the captured view depth).
                        VkClearValue vegColorClear{};
                        vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                        VkClearValue vegDepthClear{};
                        vegDepthClear.depthStencil = {1.0f, 0};
                        RendererUtils::beginColorDepthPass(vegCmd, vegColorView, vegDepthView, w, h,
                                                           vegColorClear, vegDepthClear);
                        sceneRenderer->vegetationRenderer->drawColor(this, vegCmd, vegCamPos,
                                                                     /*drawBillboards=*/false,
                                                                     /*drawImpostors=*/true);
                        vkCmdEndRendering(vegCmd);
                    }
                } else {
                    // Clear the offscreen so the composite sees empty vegetation.
                    VkClearValue vegColorClear{};
                    vegColorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                    VkClearValue vegDepthClear{};
                    vegDepthClear.depthStencil = {1.0f, 0};
                    RendererUtils::beginColorDepthPass(vegCmd, vegColorView, vegDepthView, w, h,
                                                       vegColorClear, vegDepthClear);
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

        // --- SDF debug cubes + generic SDF volumes (fire/grass) on one CB
        // (signals tlSdf) ---
        // Both passes render to their own offscreens on the dedicated
        // sdfQueue. Debug cubes depend on the cull task's SDF buffers
        // (tlCull); the generic SdfRenderer is self-contained (own grid
        // upload) and shares the same submit so only one tlSdf signal is
        // needed.
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
                // Wait on tlCull (cull buffers) and tlSolid (solid depth);
                // signal tlSdf for the composite. There is no raster water
                // pass, so tlWater is never signaled or waited on.
                std::vector<VkSemaphore> sdfWaitSemaphores = {tlCull, tlSolid};
                std::vector<uint64_t> sdfWaitValues = {v, v};
                app->submitCommandBufferAsyncToQueue(sdfCmd, app->getSdfQueue(), &tlSdf,
                    sdfWaitSemaphores, true, {}, sdfWaitValues, v, {}, true);
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

        // ── Video layer pass (Solid -> Video) ──
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
        // S2: the export walks the octree without a long-held tree lock; do not
        // start it while the content worker is mutating the tree or while the
        // chunk dispatch is still reading it.
        if (contentLoading_.load(std::memory_order_acquire) ||
            contentStreaming_.load(std::memory_order_acquire)) {
            std::cout << "[MyApp] Blender export ignored while content is loading\n";
            return;
        }
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

        // S2: while the content worker owns the octree, structural scene
        // commands are disabled and the explorer is suppressed. While chunks
        // are still streaming, structural commands stay disabled too (a new
        // build would have to join the dispatch threads on the main thread).
        // The rest of the UI (camera, settings, textures) stays interactive.
        const bool contentTreeLocked = contentLoading_.load(std::memory_order_acquire);
        const bool contentBusy = contentTreeLocked ||
            contentStreaming_.load(std::memory_order_acquire);

        // VULKAN_HIDE_UI also hides the menu bar for clean scene captures.
        static const bool hideUiMenu = std::getenv("VULKAN_HIDE_UI") != nullptr;
        if (!hideUiMenu && ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Generate Map", nullptr, false, !contentBusy))
                    generateMapPending = true;
                if (ImGui::MenuItem("Action", nullptr, false, !contentBusy)) action();
                ImGui::Separator();
                if (ImGui::MenuItem("Save Scene...", nullptr, false, !contentBusy))
                    scenePicker_.open(sceneFolderBuf, true,  "Mode: Save");
                if (ImGui::MenuItem("Load Scene...", nullptr, false, !contentBusy))
                    scenePicker_.open(sceneFolderBuf, false, "Mode: Load");
                ImGui::Separator();
                if (ImGui::MenuItem("Export to Blender (.blend)...", nullptr, false, !contentBusy))
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
            if (scenePicker_.render(chosenScene) && !contentBusy) {
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
            if (blenderPicker_.render(chosenGlb) && !contentBusy) {
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

                // Transparent / water (SDF-raymarched: no chunk meshes)
                size_t transparentLoaded = 0;
                uint32_t transparentVisible = 0;
                ImGui::Text("Transparent - Loaded (GPU): %zu  Visible (GPU cull): %u", transparentLoaded, transparentVisible);
                size_t transparentTracked = sceneRenderer ? sceneRenderer->getTransparentModelCount() : 0;
                ImGui::Text("Transparent Models Tracked: %zu", transparentTracked);

                // Vegetation
                size_t vegChunks = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getChunkCount() : 0;
                size_t vegInstances = sceneRenderer && sceneRenderer->vegetationRenderer ? sceneRenderer->vegetationRenderer->getInstanceTotal() : 0;
                ImGui::Text("Vegetation Chunks: %zu", vegChunks);
                ImGui::Text("Vegetation Instances: %zu", vegInstances);

                // ── Per-chunk GPU mesh utilization (cross-edge capacity plan) ──
                // The snapshot is versioned/cached inside SceneRenderer: ask for
                // it only while the section is expanded, and hold one shared_ptr
                // for the whole table. Re-sorting happens only after a build.
                if (sceneRenderer && ImGui::CollapsingHeader("Chunk Mesh Utilization")) {
                    const auto meshProfiles = sceneRenderer->chunkMeshProfileSnapshot();
                    if (!meshProfiles || meshProfiles->chunks.empty()) {
                        ImGui::Text("No chunk builds recorded yet.");
                    } else {
                        const ChunkMeshProfileAggregate& a = meshProfiles->aggregate;
                        ImGui::Text("Chunks %zu  built %zu  empty %zu  failed %zu",
                                    a.chunks, a.built, a.empty, a.failed);
                        ImGui::Text("Cross edges %llu  low-util %zu  overflow %zu",
                                    static_cast<unsigned long long>(a.crossEdges),
                                    a.lowUtilizationChunks, a.overflowChunks);
                        ImGui::Text("Vertex  %llu/%llu (%.1f%%)  per-chunk %.1f/%.1f/%.1f/%.1f",
                                    static_cast<unsigned long long>(a.vertexCount),
                                    static_cast<unsigned long long>(a.vertexCapacity),
                                    a.vertexUtilization,
                                    a.vertexStats.min, a.vertexStats.mean,
                                    a.vertexStats.median, a.vertexStats.max);
                        ImGui::Text("Index   %llu/%llu (%.1f%%)  per-chunk %.1f/%.1f/%.1f/%.1f",
                                    static_cast<unsigned long long>(a.indexCount),
                                    static_cast<unsigned long long>(a.indexCapacity),
                                    a.indexUtilization,
                                    a.indexStats.min, a.indexStats.mean,
                                    a.indexStats.median, a.indexStats.max);
                        ImGui::Text("Combined byte-weighted %.1f%%  per-chunk %.1f/%.1f/%.1f/%.1f",
                                    a.combinedUtilization,
                                    a.combinedStats.min, a.combinedStats.mean,
                                    a.combinedStats.median, a.combinedStats.max);
                        if (ImGui::BeginChild("##chunk-mesh-profile", ImVec2(0.0f, 220.0f), true)) {
                            constexpr ImGuiTableFlags tableFlags =
                                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_SizingStretchProp;
                            if (ImGui::BeginTable("##chunk-mesh-profile-table", 10, tableFlags)) {
                                ImGui::TableSetupColumn("Layer");
                                ImGui::TableSetupColumn("Chunk");
                                ImGui::TableSetupColumn("LoD");
                                ImGui::TableSetupColumn("Status");
                                ImGui::TableSetupColumn("CrossE");
                                ImGui::TableSetupColumn("V used/cap");
                                ImGui::TableSetupColumn("V %");
                                ImGui::TableSetupColumn("I used/cap");
                                ImGui::TableSetupColumn("I %");
                                ImGui::TableSetupColumn("Comb %");
                                ImGui::TableHeadersRow();
                                // Clip the row loop: scenes can hold thousands of
                                // chunks and the table is rebuilt every frame.
                                ImGuiListClipper clipper;
                                clipper.Begin(static_cast<int>(meshProfiles->chunks.size()));
                                while (clipper.Step()) {
                                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                                        const ChunkMeshProfile& p =
                                            meshProfiles->chunks[static_cast<size_t>(row)];
                                        ImGui::TableNextRow();
                                        ImGui::TableNextColumn(); ImGui::Text("%d", p.layer);
                                        ImGui::TableNextColumn(); ImGui::Text("%u", p.chunkId);
                                        ImGui::TableNextColumn(); ImGui::Text("%u", static_cast<unsigned>(p.lod));
                                        ImGui::TableNextColumn();
                                        ImGui::Text("%s", p.statusName());
                                        ImGui::TableNextColumn(); ImGui::Text("%u", p.crossEdgeCount);
                                        ImGui::TableNextColumn(); ImGui::Text("%u/%u", p.vertexCount, p.vertexCapacity);
                                        ImGui::TableNextColumn(); ImGui::Text("%.1f", p.vertexUtilization());
                                        ImGui::TableNextColumn(); ImGui::Text("%u/%u", p.indexCount, p.indexCapacity);
                                        ImGui::TableNextColumn(); ImGui::Text("%.1f", p.indexUtilization());
                                        ImGui::TableNextColumn();
                                        if (p.overflowed())
                                            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.2f, 1.0f), "%.1f", p.combinedUtilization());
                                        else
                                            ImGui::Text("%.1f", p.combinedUtilization());
                                    }
                                }
                                ImGui::EndTable();
                            }
                        }
                        ImGui::EndChild();
                    }
                }

              
                if (profilingEnabled) {
                    ImGui::Separator();
                    ImGui::Text("--- GPU Timing (ms) ---");
                    float gpuTotal = profileDepthPrepass + profileSky + profileSolidDraw +
                                     profileVegetationImpostor + profileWater +
                                     profileSdf + profilePostProcess + profileImGui;
                    ImGui::Text("Depth Prepass: %.2f", profileDepthPrepass);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("Solid Draw:    %.2f", profileSolidDraw);
                    ImGui::Text("Vegetation:    %.2f", profileVegetationImpostor);
                    ImGui::Text("Water:         %.2f", profileWater);
                    ImGui::Text("SDF Engine:    %.2f", profileSdf);
                    if (sceneRenderer && sceneRenderer->octreeRaymarchRenderer) {
                        ImGui::Text("Octree March:  %.2f", profileOctreeMarch);
                    }
                    ImGui::Text("PostProcess:   %.2f", profilePostProcess);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("--- GPU Total:  %.2f ---", gpuTotal);
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

        // Render all widgets. S2: the octree explorer walks raw nodes without
        // the tree lock; skip it while the content worker is building.
        // VULKAN_HIDE_UI=1 keeps the 3D scene only (headless capture/A-B).
        static const bool hideUiForCapture = std::getenv("VULKAN_HIDE_UI") != nullptr;
        if (!hideUiForCapture) {
            if (contentTreeLocked) {
                std::vector<Widget*> skip;
                if (octreeExplorerWidget) skip.push_back(octreeExplorerWidget.get());
                widgetManager.renderAllExcept(skip);
            } else {
                widgetManager.renderAll();
            }
        }

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

        // S1: while the last boot stages run, keep a compact progress readout
        // on top of the full UI. It disappears when setup completes.
        if (isLoading) drawBootPanel(true);
        // S2: content build / chunk-streaming readout, independent of the boot
        // stage. It is the visible sign that the world is being built while
        // the camera and the rest of the UI stay live.
        if (contentLoading_.load(std::memory_order_acquire) ||
            contentStreaming_.load(std::memory_order_acquire)) {
            drawContentPanel();
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
                // Water is already in the solid color/depth (octree raymarch).
                // Bind the zero-alpha dummy so the composite's water branch
                // collapses to the base color.
                sceneRenderer->waterRenderer->getDummyWaterColorView(),
                sceneRenderer->waterRenderer->getDummyWaterColorView(),
                sceneRenderer->waterRenderer->getDummyWaterColorView(),
                brushColorView,
                brushDepthView,
                sceneRenderer->waterRenderer->getDummyWaterColorView(),
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
                false, // no raster water pass: its blur inputs are dummies
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

        // ── Wind raymarch mode (WindRaymarchRenderer) ────────────────────────
        // Renders |wind| == target speed of the shared field as a raymarched
        // isosurface over the composed frame. Its parameters UBO and host->
        // fragment barrier were recorded in preRenderPass (the barrier is not
        // legal inside dynamic rendering); the march is clipped by the solid
        // scene depth. Drawn before the trails so the particle overlay stays on
        // top.
        if (sceneRenderer && sceneRenderer->windRaymarchRenderer) {
            sceneRenderer->windRaymarchRenderer->render(
                this, commandBuffer, frameIdx, settings.windRaymarchEnabled);
        }

        // ── Wind-trail particles (WindRenderer) ──────────────────────────────
        // Translucent world-space streaks over the composed frame. The
        // simulation was dispatched in preRenderPass (compute cannot be
        // recorded inside this rendering scope) and its compute->vertex barrier
        // is already recorded, so the state read here is this frame's.
        if (sceneRenderer && sceneRenderer->windRenderer) {
            sceneRenderer->windRenderer->render(
                this, commandBuffer, frameIdx, settings.windParticlesEnabled,
                settings.windParticles);
        }

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
    // S2: join the background content builder before the octrees/GPU resources
    // are destroyed. It is CPU-only, so no Vulkan call happens on the worker.
    if (contentThread.joinable()) contentThread.join();
    contentLoading_.store(false);
    contentBuildDone_.store(false);
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
        // Cancel the Blender export and join the content builder before the
        // scene/allocators are torn down.
        blenderExportCancel_.store(true);
        if (blenderExportThread_.joinable()) blenderExportThread_.join();
        if (contentThread.joinable()) contentThread.join();
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

// ── Octree SDF raymarch table (only terrain geometry path) ───────────────────

uint64_t MyApp::raymarchLayerSignature() const {
    if (!world) return 0;
    const Scene& scene = world->activeScene();
    // FNV-1a over the layer list shape. IDs are stable handles; the octree
    // pointer catches a layer re-created with the same id.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(scene.layerCount());
    for (size_t i = 0; i < scene.layerCount(); ++i) {
        const Layer* l = scene.layerAt(i);
        if (!l) {
            mix(0xffffffffu);
            continue;
        }
        mix(static_cast<uint64_t>(static_cast<uint32_t>(l->id())));
        mix(l->enabled() ? 1u : 0u);
        mix(static_cast<uint64_t>(l->renderer()));
        mix(reinterpret_cast<uintptr_t>(l->octree()));
    }
    return h;
}

void MyApp::refreshRaymarchTable(float deltaTime) {
    // S2: the table is walked from the octree. Never read it while the content
    // worker owns the tree; contentReady gates the empty pre-content state.
    if (contentLoading_.load(std::memory_order_acquire) ||
        !contentReady_.load(std::memory_order_acquire)) {
        return;
    }
    if (!sceneRenderer || !sceneRenderer->octreeRaymarchRenderer) return;
    OctreeRaymarchRenderer& renderer = *sceneRenderer->octreeRaymarchRenderer;

    // Remote scenes stream chunk records and carry no local octree SDFs, so
    // there is nothing to march. There is no raster terrain fallback.
    if (!world || useRemote) return;

    const uint64_t sig = raymarchLayerSignature();
    if (!raymarchSignaturePrimed_ || sig != raymarchLayerSignature_) {
        const bool firstBuild = !raymarchSignaturePrimed_;
        raymarchLayerSignature_ = sig;
        raymarchSignaturePrimed_ = true;
        raymarchDirty_ = true;
        // The first build after content load must be immediate (no raster
        // terrain exists to cover the debounce); layer-list changes keep the
        // short debounce so a rebuild burst coalesces.
        raymarchDirtyTime_ = firstBuild ? 1.0f : 0.0f;
    }
    if (!raymarchDirty_) return;
    // Terrain is always ray marched for local scenes; there is no raster
    // fallback to gate the table behind.
    raymarchDirtyTime_ += std::max(deltaTime, 0.0f);
    if (raymarchDirtyTime_ < 0.4f) return;

    raymarchDirty_ = false;
    raymarchDirtyTime_ = 0.0f;

    OctreeRaymarchTable rebuilt;
    world->activeScene().buildRaymarchTable(rebuilt, renderer.maxNodes());
    // Every rebuild gets a new revision, including an empty result, so stale
    // per-frame slots are overwritten on their next use.
    rebuilt.version = ++raymarchTableVersion_;
    raymarchTable_ = std::move(rebuilt);

    // Publish into this frame's slot now; the other slots version-check on
    // their next use (upload() is a no-op when the revision matches).
    renderer.upload(getCurrentFrame(), raymarchTable_);

    // Terrain has no GPU mesh, so vegetation is placed straight from the
    // surface nodes of the table that was just rebuilt.
    generateVegetationFromRaymarchTable();
}

void MyApp::generateVegetationFromRaymarchTable() {
    if (!sceneRenderer || !sceneRenderer->vegetationRenderer) return;
    VegetationRenderer* veg = sceneRenderer->vegetationRenderer.get();
    if (!veg) return;

    const OctreeRaymarchTable& table = raymarchTable_;
    if (table.nodes.empty() || table.layerRoots.empty()) {
        veg->clearAllInstances();
        return;
    }

    constexpr float kBucketSize = 512.0f;
    constexpr size_t kMaxSamples = 4'000'000;
    // Same placement rule as grass::kDensity (area-weighted slots), scaled up
    // for a denser field: the original 1-per-100-area-unit rate.
    constexpr float kDensity = 0.04f;
    constexpr uint32_t kMaxPerPatch = 2048;
    const uint64_t grassMask = sceneGrassBrushMask();

    struct Bucket {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        std::vector<int> brushes;
    };
    std::unordered_map<uint64_t, Bucket> buckets;
    buckets.reserve(1024);

    size_t totalSamples = 0;
    size_t grassPatches = 0;
    size_t sampledPatches = 0;

    // The table stores each layer's nodes as one contiguous BFS slice, so a
    // layer's root index bounds its slice (the next root starts the next
    // layer). Water roots are skipped: only solid surfaces seed blades.
    for (size_t r = 0; r < table.layerRoots.size(); ++r) {
        const OctreeRaymarchRoot& root = table.layerRoots[r];
        if (root.kind != 0) continue; // 0 = Solid
        const uint32_t begin = root.nodeIndex;
        const uint32_t end = (r + 1 < table.layerRoots.size())
            ? table.layerRoots[r + 1].nodeIndex
            : static_cast<uint32_t>(table.nodes.size());

        for (uint32_t i = begin; i < end && totalSamples < kMaxSamples; ++i) {
            if (i == root.nodeIndex) continue; // whole-domain root cube
            const OctreeRaymarchNode& rec = table.nodes[i];
            const uint32_t childMask = static_cast<uint32_t>(rec.hsv1.w + 0.5f);
            const bool leaf = (rec.meta.w & OctreeRaymarchTable::FLAG_LEAF) != 0;
            // Interior nodes are covered by their stored surface children.
            if (!leaf && childMask != 0) continue;

            const float size = rec.cubeMinSize.w;
            if (!(size > 0.0f)) continue;
            const BoundingCube cube(glm::vec3(rec.cubeMinSize), size);

            float sdf[8];
            for (int c = 0; c < 4; ++c) { sdf[c] = rec.sdf0[c]; sdf[c + 4] = rec.sdf1[c]; }
            if (SDF::eval(sdf) != SpaceType::Surface) continue;

            const glm::vec3 p = SDF::getPosition(sdf, cube);
            glm::vec3 n = SDF::getNormalFromPosition(sdf, cube, p);
            const float n2 = glm::dot(n, n);
            if (!(n2 > 1e-8f)) continue;
            n *= 1.0f / std::sqrt(n2);

            const int material = landbrush::resolveVertexMaterial(rec.meta.z, p, n);
            // Original creation rule: only blade-bearing grass-biome materials
            // seed tufts (sceneGrassBrushMask). Rock/snow/painted surfaces stay
            // bare instead of being covered uniformly.
            if (!grass::isGrassMaterial(grassMask, material)) continue;
            if (n.y < 0.35f) continue; // cliffs stay bare
            ++grassPatches;

            // One sample per ~kDensity of the patch footprint, capped so a
            // coarse flat leaf cannot monopolise the instance budget.
            const float area = size * size * glm::clamp(n.y, 0.0f, 1.0f);
            uint32_t count = static_cast<uint32_t>(
                std::lround(static_cast<double>(area) * kDensity));
            count = std::clamp(count, 1u, kMaxPerPatch);
            ++sampledPatches;

            const glm::vec3 cellMin(cube.getMin());
            const float cellMaxY = cube.getMax().y;
            const auto sdfAt = [&](float x, float y, float z) {
                return SDF::interpolate(sdf, glm::vec3(x, y, z), cube);
            };

            uint32_t rng = static_cast<uint32_t>(i * 2654435761u) ^ 0x9e3779b9u;
            for (uint32_t k = 0; k < count && totalSamples < kMaxSamples; ++k) {
                const float fx = grass::randFloat(rng);
                const float fz = grass::randFloat(rng);
                const float x = cellMin.x + fx * size;
                const float z = cellMin.z + fz * size;

                // Vertical crossing inside this surface cell: bisect the
                // trilinear field so blades sit on the real SDF surface
                // instead of the patch-centre plane.
                float lo = cellMin.y, hi = cellMaxY;
                float flo = sdfAt(x, lo, z);
                float fhi = sdfAt(x, hi, z);

                if (flo * fhi > 0.0f) continue; // no surface in this column

                for (int it = 0; it < 12; ++it) {
                    const float mid = 0.5f * (lo + hi);
                    const float fm = sdfAt(x, mid, z);
                    if (flo * fm <= 0.0f) { hi = mid; fhi = fm; }
                    else { lo = mid; flo = fm; }
                }
                const float y = 0.5f * (lo + hi);

                const uint64_t key =
                    (static_cast<uint64_t>(static_cast<uint32_t>(std::floor(x / kBucketSize))) * 73856093ull) ^
                    (static_cast<uint64_t>(static_cast<uint32_t>(std::floor(y / kBucketSize))) * 19349663ull) ^
                    (static_cast<uint64_t>(static_cast<uint32_t>(std::floor(z / kBucketSize))) * 83492791ull);
                const uint64_t nid = key != 0 ? key : 1ull;
                Bucket& bucket = buckets[nid];
                bucket.positions.emplace_back(x, y, z);
                bucket.normals.push_back(n);
                // Keep the resolved terrain material: the shared placement
                // logic applies the same dominant-vertex grass test as the
                // raster/export paths.
                bucket.brushes.push_back(material);
                ++totalSamples;
            }
        }
    }

    // Replace the previous set (layer edits rebuild the table).
    veg->clearAllInstances();
    for (auto& kv : buckets) {
        if (kv.second.positions.empty()) continue;
        veg->generateForSamples(this, kv.first, kv.second.positions,
                                kv.second.normals, kv.second.brushes);
    }

    std::cout << "[veg] raymarch placement: samples=" << totalSamples
              << " chunks=" << buckets.size()
              << " grassPatches=" << grassPatches
              << " sampledPatches=" << sampledPatches << std::endl;
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

    // Refresh the generated layers. The mixer rebuilds its compute-view
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
        case 11: // Tetrahedron
            applyPrimitive<TetrahedronDistanceFunction>(entry, model, sweepStart, minSize, callback);
            break;
        default:
            std::cerr << logPrefix << " Unknown sdfType " << entry.sdfType << ", skipping" << std::endl;
            break;
    }
}

// Shared effect+apply: wraps func in the selected effect (if any) and applies to octree.
static void applyBrushWithEffect(const BrushEntry& entry, SignedDistanceFunction& func,
                                 Octree& octree, const SignedDistanceOperation& op,
                                 const Transformation& model, const BrushPainter& brush,
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
                octree.apply(op, effect, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 1: {
                PerlinCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                // Keep the brush's UI threshold in effect while the constructor
                // stays aligned with PerlinDistortDistanceEffect.
                effect.threshold = entry.effectThreshold;
                octree.apply(op, effect, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 2: {
                SineDistortDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency, glm::vec3(0), model, minSize);
                octree.apply(op, effect, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 3: {
                VoronoiCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectCellSize,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            default:
                octree.apply(op, func, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
                break;
        }
    } else {
        octree.apply(op, func, brush, TriplanarMappingPainter(), minSize, simplifier, updateHandler, deleteHandler);
    }
}

// Bounds of one brush application (union of the current and, in sweep mode,
// the previous placement), padded for the optional effect displacement. Shared
// Bounds for one primitive (or its sweep trail), using the same fixed
// non-sweep placement the GPU preview sees. `sweepStart` is ignored when
// `entry.sweepMode` is off.
static BoundingSphere brushEntryBounds(const BrushEntry& entry, const glm::vec3& sweepStart) {
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
}

static BoundingSphere unionBoundingSpheres(const BoundingSphere& a, const BoundingSphere& b) {
    const glm::vec3 delta = b.center - a.center;
    const float dist = glm::length(delta);
    if (dist + b.radius <= a.radius) return a;
    if (dist + a.radius <= b.radius) return b;
    const float radius = 0.5f * (dist + a.radius + b.radius);
    const glm::vec3 center = (dist > 1e-6f)
        ? a.center + delta * ((radius - a.radius) / dist)
        : a.center;
    return BoundingSphere(center, radius);
}

// by the GPU preview and the undo history intersection tests.
static void computeBrushBounds(const BrushEntry& entry, const glm::vec3& sweepStart,
                               glm::vec3& boundsCenter, float& boundsRadius) {
    BoundingSphere bounds = brushEntryBounds(entry, sweepStart);
    if (entry.useInterpolation) {
        const BrushEntry target = brushEntryFromTarget(entry.interpolationTarget);
        const glm::vec3 targetSweepStart = target.sweepMode ? target.previousTranslate
                                                            : target.translate;
        bounds = unionBoundingSpheres(bounds, brushEntryBounds(target, targetSweepStart));
    }
    if (entry.useEffect) {
        float pad = entry.effectAmplitude * 1.97f;
        if (entry.effectType == 2) pad = entry.effectAmplitude * 0.5f;
        else if (entry.effectType == 3) pad = std::abs(entry.effectAmplitude * entry.effectContrast);
        bounds.radius += pad;
    }
    boundsRadius = std::max(bounds.radius, 0.1f) * 1.05f;
    boundsCenter = bounds.center;
}

// Per-frame brush preview update: stream the selected entry's SDF
// description into the BrushSdfRenderer's UBO (consumed by the preview
// raymarcher and by the solid shader's PAINT/REMOVE intersection test). No
// octree and no tessellation: moving/scaling the brush is just a UBO rewrite.
void MyApp::applyBenchCameraPath() {
    if (benchCameraPath_ == BenchCameraPath::None) return;

    // Hold the start pose while content is building, then advance from frame
    // zero of the ready period. This keeps two A/B runs aligned even when
    // their load times differ.
    if (!contentReady_.load(std::memory_order_acquire) ||
        contentLoading_.load(std::memory_order_acquire)) {
        camera.setPose(benchCameraBasePos_, benchCameraBaseOri_);
        return;
    }

    // Frame-indexed so an A/B pair sees the same pose at the same frame,
    // independent of wall-clock timing. Nominal cadence: 60 fps.
    const float frame = static_cast<float>(benchCameraFrame_++);
    glm::vec3 pos = benchCameraBasePos_;
    glm::quat ori = benchCameraBaseOri_;
    const glm::vec3 forward = glm::normalize(glm::rotate(
        benchCameraBaseOri_, glm::vec3(0.0f, 0.0f, -1.0f)));
    glm::vec3 right = glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f));
    right = (glm::dot(right, right) > 1e-8f) ? glm::normalize(right)
                                             : glm::vec3(1.0f, 0.0f, 0.0f);
    switch (benchCameraPath_) {
        case BenchCameraPath::Static:
            break;
        case BenchCameraPath::Pan5:
        case BenchCameraPath::Pan60: {
            const float rate = (benchCameraPath_ == BenchCameraPath::Pan5) ? 5.0f : 60.0f;
            const float yaw = glm::radians(rate) * (frame / 60.0f);
            ori = glm::normalize(glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
                                 benchCameraBaseOri_);
            break;
        }
        case BenchCameraPath::Dolly:
            pos += forward * frame;           // 1 m/frame
            break;
        case BenchCameraPath::Strafe:
            pos += right * (0.5f * frame);    // 0.5 m/frame
            break;
        case BenchCameraPath::None:
            break;
    }
    camera.setPose(pos, ori);
}

void MyApp::updateBrushPreview(uint32_t frameIndex) {
    if (!sceneRenderer || !sceneRenderer->brushRenderer) return;

    BrushEntry* entry = brushManager.getSelectedEntry();
    if (!entry) {
        if (brushInterpolationWidget) {
            if (brushInterpolationWasEnabled_) brushInterpolationWidget->hide();
            brushInterpolationWasEnabled_ = false;
        }
        // No brush: zero-radius bounds disable both the preview and the
        // solid PAINT/REMOVE evaluation.
        sceneRenderer->brushRenderer->updateParams(frameIndex, BrushEntry{}, glm::vec3(0.0f),
                                                   glm::vec3(0.0f), 0.0f,
                                                   getWidth(), getHeight());
        return;
    }

    // Open/close the target-SDF window on interpolation transitions.
    if (brushInterpolationWidget) {
        if (entry->useInterpolation && !brushInterpolationWasEnabled_)
            brushInterpolationWidget->show();
        else if (!entry->useInterpolation && brushInterpolationWasEnabled_)
            brushInterpolationWidget->hide();
    }
    brushInterpolationWasEnabled_ = entry->useInterpolation;

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
    const BrushSdfTarget& target = entry->interpolationTarget;
    cachedTargetSweepStart_ = target.sweepMode ? target.previousTranslate : target.translate;
    const glm::vec3 targetSweepStart = cachedTargetSweepStart_;

    // Bounding sphere: union of the current and (for sweep mode) start
    // placements, each evaluated as a non-sweep primitive with its matching
    // model so capsule-style getSphere(model) calls see the right translate.
    // (SweepSignedDistanceFunction::getSphere reuses the outer model for both
    // ends, which collapses the trail for capsule-likes — so it is not used
    // here.) Padded for the optional effect displacement like the CPU effect
    // constructors (bias += amplitude * factor). The target's current/sweep
    // bounds are merged when interpolation is enabled.
    glm::vec3 boundsCenter(entry->translate);
    float boundsRadius = glm::length(entry->scale);
    computeBrushBounds(*entry, cachedSweepStart, boundsCenter, boundsRadius);

    sceneRenderer->brushRenderer->updateParams(frameIndex, *entry, cachedSweepStart,
                                               boundsCenter, boundsRadius,
                                               getWidth(), getHeight(), targetSweepStart);

    // Advance the sweep start for the next frame's trail.
    if (entry->sweepMode) entry->previousTranslate = entry->translate;
    if (entry->useInterpolation && entry->interpolationTarget.sweepMode) {
        entry->interpolationTarget.previousTranslate = entry->interpolationTarget.translate;
    }
}

void MyApp::applyBrushEventToLayer(const BrushEntry& entry, const glm::vec3& sweepStart, Layer::Id targetLayer) {
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
    Layer* targetLayerPtr = world->scene().findLayer(targetLayer);
    Octree* octreePtr = targetLayerPtr ? targetLayerPtr->octree() : nullptr;
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
            svgBrush = std::make_unique<SvgBrush>(entry.materialIndex, entry.hsv);
        }
    }
    const BrushPainter& brush =
        svgBrush ? static_cast<const BrushPainter&>(*svgBrush)
                 : static_cast<const BrushPainter&>(simpleBrush);

    Simplifier simplifier(0.05f);
    // The tetrahedron's sampled field needs a finer octree lattice than the
    // widget default to represent its beveled edges; other primitives keep the
    // user's Min Size unchanged.
    BrushEntry sampled = entry;
    sampled.minSize = brushSdfSampleMinSize(entry.sdfType, entry.scale, entry.minSize);
    auto applyEntry = [&](SignedDistanceFunction& sourceFunc) {
        if (!sampled.useInterpolation) {
            applyBrushWithEffect(sampled, sourceFunc, octree, brushOp, model, brush, simplifier,
                                 updateHandler, deleteHandler);
            return;
        }

        BrushEntry target = brushEntryFromTarget(sampled.interpolationTarget);
        target.minSize = brushSdfSampleMinSize(target.sdfType, target.scale, target.minSize);
        const Transformation targetModel(target.scale, target.translate, target.rot);
        const glm::vec3 targetSweepStart = target.sweepMode ? target.previousTranslate
                                                            : target.translate;
        bool mixedApplied = false;
        auto applyTarget = [&](SignedDistanceFunction& targetFunc) {
            InterpolatedDistanceFunction mixed(sourceFunc, targetFunc,
                                               sampled.interpolationFactor, model, sampled.minSize);
            applyBrushWithEffect(sampled, mixed, octree, brushOp, model, brush, simplifier,
                                 updateHandler, deleteHandler);
            mixedApplied = true;
        };
        forEachBrushSDF(target, targetModel, targetSweepStart, target.minSize,
                        "[applyBrushEventToLayer:target]", applyTarget);
        if (!mixedApplied) {
            // Invalid target SDF (e.g. an unreadable SVG): keep the source.
            applyBrushWithEffect(sampled, sourceFunc, octree, brushOp, model, brush, simplifier,
                                 updateHandler, deleteHandler);
        }
    };
    forEachBrushSDF(sampled, model, sweepStart, sampled.minSize, "[applyBrushEventToLayer]", applyEntry);

    // A brush edit changes stored SDF corners even for chunks whose
    // tessellation batch is not drained this frame (e.g. edits outside the
    // current LoD/visibility window). Mark the raymarch table dirty here, at
    // the single place every paint/undo application funnels through, so the
    // debounced rebuild never depends on mesh-upload timing. The timer is set
    // past the debounce: a paint stroke is interactive, so the very next
    // refreshRaymarchTable() must rebuild instead of waiting 0.4 s.
    raymarchDirty_ = true;
    raymarchDirtyTime_ = 1.0f;
}

void MyApp::ensureBrushBase(Layer::Id layer, Octree& tree) {
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

void MyApp::recordBrushEvent(Layer::Id layer, const BrushHistoryEvent& event) {
    if (layer < 0) return;
    if (brushHistories.size() <= static_cast<size_t>(layer)) brushHistories.resize(static_cast<size_t>(layer) + 1);
    brushHistories[static_cast<size_t>(layer)].events.push_back(event);
}

void MyApp::undoBrushLayer(Layer::Id layer, bool group) {
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
    Layer* layerPtr = world->scene().findLayer(layer);
    Octree* tree = layerPtr ? layerPtr->octree() : nullptr;
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
            sceneRenderer->processPendingMeshes(this, pending);
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
        if (layer < static_cast<Layer::Id>(collectors.size()) && collectors[layer]) collectors[layer]->clear();
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
        Layer* l = scene.findLayer(layer);
        if (l) scene.invalidateTessellationCache(*l);
        Octree::OctreeNodeDataHandler updateHandler;
        {
            std::lock_guard<std::mutex> lock(layersSyncMutex_);
            if (layer >= 0 && static_cast<size_t>(layer) < collectors.size() && collectors[layer]) {
                updateHandler = collectors[layer]->updateHandler;
            }
        }
        if (updateHandler && l) {
            scene.reemitLayer(*l, updateHandler);
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
    // S2: undo replays brush events into the octree; defer while content builds.
    if (contentLoading_.load(std::memory_order_acquire)) return;
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
            if (entry) undoBrushLayer(entry->targetLayer, shift);
        }
    }
    undoKeyWasDown = combo;
}

void MyApp::applyBrushToScene() {
    if (!world || !sceneRenderer) return;
    // S2: never mutate the tree while the content worker owns it.
    if (contentLoading_.load(std::memory_order_acquire)) return;
    // Brush edits target the LOCAL octrees; in remote mode there is nothing
    // local to edit (the server owns the SDF), so refuse instead of silently
    // building invisible local content.
    if (useRemote) {
        std::cout << "[MyApp::applyBrushToScene] ignored in remote mode (server owns the scene)\n";
        return;
    }

    const BrushEntry* selectedEntry = brushManager.getSelectedEntry();
    if (!selectedEntry) return;
    BrushEntry entry = *selectedEntry;
    if (entry.useInterpolation) {
        // Use the same start placement the preview captured this frame.
        entry.interpolationTarget.previousTranslate = cachedTargetSweepStart_;
    }
    const Layer::Id targetLayer = entry.targetLayer;
    if (!world->activeScene().validLayer(targetLayer)) {
        std::cerr << "[MyApp::applyBrushToScene] invalid targetLayer " << entry.targetLayer << std::endl;
        return;
    }

    if (!undoReplaying) {
        Layer* targetLayerPtr = world->scene().findLayer(targetLayer);
        if (Octree* tree = targetLayerPtr ? targetLayerPtr->octree() : nullptr) {
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
    if (entry.useInterpolation && entry.interpolationTarget.sweepMode) {
        BrushEntry* mutableEntry = brushManager.getSelectedEntry();
        if (mutableEntry) {
            mutableEntry->interpolationTarget.previousTranslate =
                mutableEntry->interpolationTarget.translate;
        }
    }
}

// Ensure pending texture generation requests are flushed after a frame is submitted
// so array-layer transitions happen outside of active draw command buffers.

void MyApp::action() {
    // S2: the content worker owns the octree; a re-tessellation must wait.
    if (contentLoading_.load(std::memory_order_acquire)) {
        std::cout << "[MyApp::action] ignored while content is loading\n";
        return;
    }
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
    // S2: the content worker owns the octrees while it builds. Join it before
    // any GPU/CPU scene teardown so it can never touch recycled nodes.
    if (contentThread.joinable()) contentThread.join();
    contentLoading_.store(false);
    contentBuildDone_.store(false);
    contentStreaming_.store(false);
    contentReady_.store(false);
    contentChunksExpected_.store(0);
    contentError_.clear();
    // The Blender export walks the octree: stop it before resetting the tree.
    if (blenderExportRunning_.load()) {
        blenderExportCancel_.store(true);
        if (blenderExportThread_.joinable()) blenderExportThread_.join();
        blenderExportRunning_.store(false);
        blenderExportFinished_.store(false);
    }
    // The scene server holds references into the live scene.
    if (sceneServer) {
        sceneServer->stop();
        sceneServer.reset();
    }
    stopRemoteDispatch();
    if (sceneProcessThread.joinable()) sceneProcessThread.join();
    sceneDispatchRunning_.store(false);
    deviceWaitIdle();
    processPendingCommandBuffers();

    // S2: the previous scene's raymarch table references copied nodes that no
    // longer exist. Clear it now (device idle) so the old isosurface does not
    // linger while the new content builds.
    if (sceneRenderer && sceneRenderer->octreeRaymarchRenderer) {
        raymarchTable_.clear();
        for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
            sceneRenderer->octreeRaymarchRenderer->upload(f, raymarchTable_);
    }
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

    // The content tree is about to change: its droplet collision table can no
    // longer be valid. Rebuild it in finalizeContentLoad() after the build.
    if (particleSim) {
        particleSim->destroy(this);
        particleSim.reset();
        dropletDrawCount = 0;
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

void MyApp::dispatchLayerEvents(Layer::Id layer) {
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
    if (world) {
        const Layer* l = world->activeScene().findLayer(layer);
        if (!l || !l->enabled()) {
            collector->clear();
            return;
        }
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
    // Snapshot the live layer ids; per-layer dispatch re-locks to copy.
    std::vector<Layer::Id> ids;
    if (world) {
        Scene& scene = world->activeScene();
        ids.reserve(scene.layerCount());
        for (size_t i = 0; i < scene.layerCount(); ++i)
            if (const Layer* l = scene.layerAt(i)) ids.push_back(l->id());
    }
    for (Layer::Id id : ids) dispatchLayerEvents(id);
}

void MyApp::ensureLayerHandlers() {
    if (!world || !sceneRenderer) return;
    std::lock_guard<std::mutex> lock(layersSyncMutex_);
    Scene& scene = world->activeScene();
    // Live ids (stable, never reused) and the vector size that covers them.
    std::vector<Layer::Id> ids;
    ids.reserve(scene.layerCount());
    Layer::Id maxId = 1;
    for (size_t i = 0; i < scene.layerCount(); ++i) {
        if (const Layer* l = scene.layerAt(i)) {
            ids.push_back(l->id());
            maxId = std::max(maxId, l->id());
        }
    }
    const size_t n = static_cast<size_t>(maxId) + 1;
    sceneRenderer->ensureLayerStates(n);
    if (ids == knownLayerIds_ && collectors.size() == n &&
        layerAddHandlers.size() == n && layerRemoveHandlers.size() == n)
        return;
    knownLayerIds_ = ids;
    // Per-id dispatch arrays. Holes are removed layers: keep null collector
    // entries so a queued id can be dropped without touching another layer.
    collectors.resize(n);
    for (size_t i = 0; i < n; ++i)
        if (!collectors[i]) collectors[i] = std::make_shared<UniqueChangeCollector>();
    layerAddHandlers.assign(n, {});
    layerRemoveHandlers.assign(n, {});
    // Rebuild the handlers for every live layer: build() captures
    // PublishTarget references to the layer's heap-allocated GPU state, so a
    // removed layer's handlers must not stay reachable.
    Scene* sceneForChanges = &world->activeScene();
    for (Layer::Id id : ids) {
        auto handlers = build(
            sceneRenderer, this, sceneForChanges, id,
            &sceneRenderer->layerGenPool(id),
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->layerChunks(id),
                sceneRenderer->layerMutex(id),
                sceneRenderer->layerPendingDeletes(id)
            });
        layerAddHandlers[id] = handlers.first;
        layerRemoveHandlers[id] = handlers.second;
    }
    // Drop queued events for ids that no longer exist.
    for (size_t i = 0; i < n; ++i) {
        if (std::find(ids.begin(), ids.end(), static_cast<Layer::Id>(i)) == ids.end() && collectors[i])
            collectors[i]->clear();
    }
}

void MyApp::syncLayers() {
    // S2: the content worker owns the trees and the file path can replace the
    // layer list. Do not rebuild handlers or re-point widgets until it is done.
    if (contentLoading_.load(std::memory_order_acquire)) return;
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
        layersWidget->setAllowStructureEdit(
            !useRemote && !contentStreaming_.load(std::memory_order_acquire));
    }
    if (brush3dWidget) brush3dWidget->setScene(active);
    if (octreeExplorerWidget) octreeExplorerWidget->setScene(active);
    ensureLayerHandlers();
}

void MyApp::repopulateLayer(Layer::Id layer) {
    if (!world || !sceneRenderer) return;
    // S2: re-emitting walks the octree; wait for the content build to finish.
    if (contentLoading_.load(std::memory_order_acquire)) return;
    ensureLayerHandlers();
    Scene& scene = world->activeScene();
    Layer* layerPtr = scene.findLayer(layer);
    if (!layerPtr || !layerPtr->enabled()) return;
    // The GPU slots were freed; the octree still holds the data but the
    // tessellation dedup cache considers it up-to-date, so invalidate first
    // (otherwise the re-emit below is skipped as a no-op).
    scene.invalidateTessellationCache(*layerPtr);
    Octree::OctreeNodeDataHandler updateHandler;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        if (layer < 0 || static_cast<size_t>(layer) >= collectors.size() || !collectors[layer])
            return;
        updateHandler = collectors[layer]->updateHandler;
    }
    scene.reemitLayer(*layerPtr, updateHandler);
    // Dispatch on a worker thread (same pattern as startTessellationThreads):
    // the collector replay enqueues tessellation onto the layer's gen pool
    // and returns quickly; the main thread uploads results progressively.
    // Detached: dispatchLayerEvents only holds shared_ptr copies, so a
    // concurrent add/remove cannot free anything under it.
    std::thread([this, layer]() { dispatchLayerEvents(layer); }).detach();
}

void MyApp::applyDebugOverlaySettings() {
    if (!sceneRenderer) return;
    // S2: a toggle-on runs a tree walk (recollectDebugOverlays); never walk the
    // octree while the content worker holds its write lock.
    if (contentLoading_.load(std::memory_order_acquire)) return;
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
        Layer* layerPtr = scene.layerAt(i);
        if (!layerPtr || !layerPtr->enabled()) continue;
        const Layer::Id layer = layerPtr->id();

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
        scene.reemitLayer(*layerPtr, [&nodes](const OctreeNodeData& nd) { nodes.push_back(nd); });
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
    Layer* dropletLayer = world->scene().findLayer(LAYER_OPAQUE);
    if (!dropletLayer) {
        std::cerr << "[MyApp] droplet simulation disabled: no opaque layer\n";
        return;
    }
    const glm::vec3 rootMin = world->scene().lodRootMin(*dropletLayer);
    const glm::vec3 rootMax = rootMin + glm::vec3(world->scene().lodRootSize(*dropletLayer));
    dropletRegionMin = glm::vec3(-lateral, rootMin.y, -lateral);
    dropletRegionMax = glm::vec3(lateral, rootMax.y, lateral);

    std::vector<OctreeGpuNode> nodes;
    if (!world->scene().buildSdfRegionTable(*dropletLayer, dropletRegionMin, dropletRegionMax, nodes,
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

// S2: start the CPU-only content build on its own thread. The main thread has
// already reset the scene/GPU state; from here on frames keep presenting while
// the octree is built or loaded. postSubmit() polls contentBuildDone_ and
// calls finalizeContentLoad().
void MyApp::startLocalContentLoad(bool fromFile, std::string path) {
    if (useRemote) {
        std::cout << "[MyApp] local content load ignored in remote mode\n";
        return;
    }
    if (contentLoading_.load()) {
        std::cout << "[MyApp] content load already running; request ignored\n";
        return;
    }
    if (contentStreaming_.load(std::memory_order_acquire)) {
        std::cout << "[MyApp] content still streaming; new load ignored\n";
        return;
    }
    if (fromFile) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) {
            std::cerr << "[MyApp] loadSceneFromFile: cannot open '" << path << "'\n";
            contentError_ = "cannot open " + path;
            contentBuildFailed_.store(true);
            contentReady_.store(false);
            return;
        }
    }

    // Main-thread reset: joins old workers, waits the GPU, clears meshes and
    // resets the CPU trees. Nothing below this point touches Vulkan.
    resetSceneState();
    ensureLayerHandlers();
    if (layersWidget) layersWidget->setAllowStructureEdit(false);

    std::vector<Octree::OctreeNodeDataHandler> ups, dels;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        ups.reserve(collectors.size());
        dels.reserve(collectors.size());
        for (auto& c : collectors) {
            ups.push_back(c->updateHandler);
            dels.push_back(c->deleteHandler);
        }
    }

    const int videoIdx = videoCubeMaterialIndex;
    const int sdfIdx = sdfGivenMaterialIndex;
    contentIsFile_ = fromFile;
    contentPath_ = std::move(path);
    contentStart_ = std::chrono::steady_clock::now();
    contentError_.clear();
    contentBuildFailed_.store(false);
    contentBuildDone_.store(false);
    contentChunksExpected_.store(0);
    contentStreaming_.store(false);
    contentHasSettings_ = false;
    if (fromFile) contentSettingsCopy_ = settings;
    contentLoading_.store(true, std::memory_order_release);

    try {
        contentThread = std::thread(
        [this, fromFile, path = contentPath_, ups = std::move(ups),
         dels = std::move(dels), videoIdx, sdfIdx]() mutable {
            bool ok = true;
            try {
                if (fromFile) {
                    Settings loaded = contentSettingsCopy_;
                    world->scene().load(path, ups, dels, &loaded);
                    contentSettingsCopy_ = loaded;
                    contentHasSettings_ = true;
                    std::cout << "[MyApp::loadSceneFromFile] Octree loaded from '"
                              << path << "'\n";
                } else if (std::getenv("VULKAN_DISPLACEMENT_TEST") != nullptr) {
                    DisplacementTestLoader loader;
                    world->scene().loadScene(loader, ups, dels);
                    std::cout << "[MyApp::generateMap] displacement test scene constructed\n";
                } else {
                    MainSceneLoader loader;
                    loader.setVideoBrushIndex(videoIdx);
                    loader.setSdfGivenBrushIndex(sdfIdx);
                    world->scene().loadScene(loader, ups, dels);
                    std::cout << "[MyApp::generateMap] Octree construction complete\n";
                }
            } catch (const std::exception& e) {
                ok = false;
                contentError_ = e.what();
                std::cerr << "[MyApp] content build failed: " << contentError_ << "\n";
            } catch (...) {
                ok = false;
                contentError_ = "unknown exception";
                std::cerr << "[MyApp] content build failed (unknown exception)\n";
            }
            contentBuildFailed_.store(!ok, std::memory_order_relaxed);
            contentBuildDone_.store(true, std::memory_order_release);
        });
    } catch (const std::exception& e) {
        contentLoading_.store(false);
        contentBuildFailed_.store(true);
        contentError_ = std::string("thread creation failed: ") + e.what();
        std::cerr << "[MyApp] " << contentError_ << "\n";
    }
}

void MyApp::startSceneServerIfEnabled() {
    if (!settings.serveScene || sceneServer || !world) return;
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
}

// Main thread, after present: join the builder, apply any loaded settings,
// start tessellation and build the content-dependent GPU state (droplets).
void MyApp::finalizeContentLoad() {
    if (!contentBuildDone_.exchange(false, std::memory_order_acquire)) return;
    if (contentThread.joinable()) contentThread.join();
    contentLoading_.store(false);

    if (contentBuildFailed_.load(std::memory_order_relaxed)) {
        std::cerr << "[MyApp] content load failed: " << contentError_ << "\n";
        contentReady_.store(false);
        contentStreaming_.store(false);
        if (layersWidget) layersWidget->setAllowStructureEdit(!useRemote);
        return;
    }

    if (contentHasSettings_) {
        settings = contentSettingsCopy_;
        contentHasSettings_ = false;
    }

    contentReady_.store(true, std::memory_order_release);
    // New octree content is the ONE geometry source now (terrain does not
    // mesh), so skip the debounce: the first post-load update rebuilds and
    // uploads the raymarch table immediately.
    raymarchDirty_ = true;
    raymarchDirtyTime_ = 1.0f;

    // Expected chunk events are known before dispatch consumes the collectors.
    size_t expected = 0;
    {
        std::lock_guard<std::mutex> lock(layersSyncMutex_);
        for (auto& c : collectors) {
            if (c) expected += c->size();
        }
    }
    contentChunksExpected_.store(
        static_cast<uint32_t>(std::min<size_t>(expected, UINT32_MAX)),
        std::memory_order_relaxed);

    // Start the server before dispatch so it observes the first chunk events.
    startSceneServerIfEnabled();

    // Every layer tessellates on its own thread.
    startTessellationThreads(contentIsFile_
        ? "[MyApp::loadSceneFromFile] Scene tessellation complete\n"
        : "[MyApp::generateMap] Scene chunk tessellation complete\n");

    // Content-dependent GPU state now has a valid octree to read.
    ensureDroplets();
    contentStreaming_.store(contentChunksExpected_.load(std::memory_order_relaxed) > 0,
                            std::memory_order_relaxed);

    // Keep structural editing disabled until the stream finishes; adding or
    // removing layers while the dispatch threads read the trees is unsafe.
    if (layersWidget) layersWidget->setAllowStructureEdit(!useRemote && !contentStreaming_.load());

    if (sceneRenderer) sceneRenderer->logMemoryUtilization();
    std::cout << "[MyApp] content ready: expected chunk events="
              << contentChunksExpected_.load(std::memory_order_relaxed)
              << (contentIsFile_ ? " (file)" : " (generated)") << "\n";
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
        contentReady_.store(false, std::memory_order_relaxed);
        generateMapPending = true;
        return;
    }
    std::cout << "[MyApp::loadRemoteScene] snapshot replayed; starting live dispatch\n";
    contentReady_.store(true, std::memory_order_relaxed);
    startRemoteDispatchLoop();
}

void MyApp::postSubmit() {
    // ── S2: content build first ─────────────────────────────────────────
    // The octree build/load runs on contentThread. A completed build is
    // finalized here (after present) so its one-time wiring (tessellation
    // start, droplets, settings) never splits a frame. New requests start
    // only when no build is in flight.
    if (contentLoading_.load(std::memory_order_acquire) &&
        contentBuildDone_.load(std::memory_order_acquire)) {
        finalizeContentLoad();
    }
    if (!contentLoading_.load(std::memory_order_acquire) &&
        !contentStreaming_.load(std::memory_order_acquire)) {
        if (generateMapPending) {
            generateMapPending = false;
            startLocalContentLoad(false, std::string());
        } else if (loadScenePending) {
            loadScenePending = false;
            startLocalContentLoad(true, pendingLoadPath);
        }
    }

    // H3 (perf report 29): follow the overlay settings every frame. The widget
    // writes settings.showSDFDebug / showBoundingBoxes directly, so this is the
    // single place that turns them into collection gates (a toggle-on also runs
    // the one-shot re-collection). Steady state is four relaxed atomic loads.
    // S2: the recollect walk reads the octree; do not run it while the content
    // worker holds the tree's write lock.
    if (!contentLoading_.load(std::memory_order_acquire)) applyDebugOverlaySettings();
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
    // S2: never build it before the octree content exists (the old ordering
    // built a one-node table against an empty tree and never refreshed it).
    if (contentReady_.load(std::memory_order_acquire) &&
        !contentLoading_.load(std::memory_order_acquire) &&
        settings.dropletsEnabled && sceneRenderer &&
        (!particleSim || !particleSim->ready())) {
        ensureDroplets();
    }

    // Scene server: start/stop with the settings toggle (or VULKAN_SERVE=1).
    // Only a LocalScene can be served; a remote scene already has a server.
    // S2: start it only after the content tree is final.
    if (contentReady_.load(std::memory_order_acquire) &&
        !contentLoading_.load(std::memory_order_acquire) &&
        settings.serveScene && !sceneServer && world) {
        startSceneServerIfEnabled();
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
            const size_t waterLoaded = 0; // water is SDF-raymarched
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
            if (std::getenv("VULKAN_CHUNK_PROFILE") != nullptr)
                sceneRenderer->logChunkMeshProfiles();
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

    if (brushApplyToScenePending && !contentLoading_.load(std::memory_order_acquire)) {
        brushApplyToScenePending = false;
        applyBrushToScene();
        // The raymarch table was refreshed earlier in update(), before this
        // deferred apply. Rebuild and upload it now so the frame we are about
        // to record contains the stroke instead of the previous frame's SDF
        // (the raymarched surface used to lag one frame behind the brush).
        if (sceneRenderer && sceneRenderer->octreeRaymarchRenderer) {
            refreshRaymarchTable(0.0f);
            sceneRenderer->octreeRaymarchRenderer->upload(getCurrentFrame(), raymarchTable_);
        }
    }

    if (remoteLoadPending && !contentLoading_.load(std::memory_order_acquire)) {
        remoteLoadPending = false;
        loadRemoteScene();
        if (sceneRenderer) sceneRenderer->logMemoryUtilization();
    }

    // S2: the streaming indicator lives until every layer worker has finished
    // dispatching and the main thread has drained the pending publish queue.
    if (contentStreaming_.load(std::memory_order_relaxed) &&
        !sceneDispatchRunning_.load(std::memory_order_acquire) &&
        sceneRenderer && sceneRenderer->getPendingMeshCount() == 0) {
        contentStreaming_.store(false, std::memory_order_relaxed);
        if (layersWidget) layersWidget->setAllowStructureEdit(!useRemote);
    }
}
