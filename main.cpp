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
#include <array>
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
#include "vulkan/VulkanApp.hpp"
#include "vulkan/renderer/SceneRenderer.hpp"
#include "vulkan/renderer/RendererUtils.hpp"
#include "utils/LocalScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/SkySettings.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderTargetsWidget.hpp"
#include "widgets/BillboardCreator.hpp"
#include "widgets/ImpostorWidget.hpp"
#include "services/ImpostorService.hpp"
#include "widgets/TextureMixerWidget.hpp"
#include "widgets/TextureViewerWidget.hpp"
#include "widgets/CameraWidget.hpp"
#include "events/ControllerManager.hpp"
#include "widgets/ControllerParametersWidget.hpp"
#include "widgets/GamepadWidget.hpp"
#include "widgets/LightWidget.hpp"
#include "widgets/VulkanResourcesManagerWidget.hpp"
#include "widgets/VegetationAtlasEditor.hpp"
#include "widgets/WindWidget.hpp"
#include "widgets/OctreeExplorerWidget.hpp"
#include "widgets/Brush3dWidget.hpp"
#include "widgets/MusicWidget.hpp"
#include "widgets/components/FilePicker.hpp"
#include "sdf/AddSignedDistanceOperation.hpp"
#include "sdf/DeleteSignedDistanceOperation.hpp"
#include "sdf/PaintSignedDistanceOperation.hpp"
#include "sdf/SweepSignedDistanceFunction.hpp"
#include "utils/MainSceneLoader.hpp"
#include "space/UniqueChangeCollector.hpp"
#include "utils/Settings.hpp"
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
#include "events/RadialMenuHandler.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/MaterialManager.hpp"
#include "world/World.hpp"
#include "vulkan/renderer/DescriptorWriter.hpp"
#include "utils/BillboardManager.hpp"
#include "utils/AtlasManager.hpp"
#include "services/TextureMixer.hpp"
#include "services/BillboardService.hpp"
#include "utils/ShadowParams.hpp"
#include "space/ThreadPool.hpp"
#include "space/Octree.hpp"

// Build the {onAdded, onDeleted} renderer lambdas for one space. The main
// scene drives the ChunkManager state machine and SDF debug markers; the
// brush scene routes geometry to the separate brush queue and chunk maps
// instead. Returns the pair of renderer-side lambdas; the caller wires them
// behind a UniqueChangeCollector dedup stage and dispatches on the main
// thread.
//
// Everything that differs between a space (solid vs water, main vs brush)
// lives in this one struct so build() has no space-type branching — the four
// call sites below only fill in a few fields each.
struct PublishTarget {
    // Where finished meshes are queued for main-thread GPU upload. ONE shared
    // queue for every stream (main solid/water + brush solid/water); each
    // entry is keyed by its emitting octree node id and carries its own
    // single LoDMesh (no ladder structures anywhere).
    std::unordered_map<NodeID, SceneRenderer::PendingMeshData>& meshData;
    std::mutex& queueMutex;
    // Chunk registry whose entries are removed on delete (solid/transparent vs brush).
    std::unordered_map<NodeID, Model3DVersion>& chunks;
    // Mutex guarding [chunks] (per-space, chosen by the caller).
    std::recursive_mutex& chunksMutex;
    // IndirectRenderer owning the meshes (removeMesh/removeMeshSlotted).
    IndirectRenderer& indirect;
    // Deferred slot-registry on delete (solid/water) — only used when
    // chunkManaged + slotted mode.
    std::unordered_map<NodeID, SceneRenderer::PendingDeleteEntry>& deferredSlots;
    // True → main scene: ChunkManager state machine + SDF debug markers +
    // slot deferral on delete. False → brush scene: dedicated brush maps / IR.
    bool chunkManaged;
};

std::pair<Octree::OctreeNodeDataHandler, Octree::OctreeNodeDataHandler> build(SceneRenderer* renderer, VulkanApp* app, Scene* scene,
              Layer layer, float minSize, ThreadPool* genPool, const PublishTarget& target) {

    Octree::OctreeNodeDataHandler onAdded = [renderer, app, scene, layer, minSize, genPool, target](const OctreeNodeData& nd) {
        NodeID nid = reinterpret_cast<NodeID>(nd.node);
        ChunkManager::ChunkId cid = static_cast<ChunkManager::ChunkId>(nid);

        if (target.chunkManaged) {
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
        }

        OctreeNodeData nodeCopy = nd;
        // Single-mesh handler: each chunk emits exactly one LoDMesh and it is
        // queued as its own entry — the consumer publishes it into the chunk's
        // stable slot.
        renderer->processNodeLayer(*scene, layer, nid, nodeCopy,
            [renderer, cid, nodeCopy, target](Layer layer_, NodeID nid_, const Octree::LoDMesh& lodMesh) {
                if (lodMesh.geom.vertices.empty() || lodMesh.geom.indices.empty()) {
                    return; // no surface: nothing to publish
                }
                // lodMesh.lod is the 0-based band level (chunkLod - 1); the
                // stored chunkLod is 1-based. finishBuild only when the mesh is
                // the added node's own rung.
                if (target.chunkManaged && nodeCopy.node->getChunkLod() == lodMesh.lod) {
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
                // last tessellation result wins). One shared queue for every
                // stream — each entry is tagged brush vs main.
                std::lock_guard<std::mutex> lock(target.queueMutex);
                target.meshData[nid_] = {layer_, nid_, lodMesh, nodeCopy, /*isBrush=*/!target.chunkManaged};
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

        if (target.chunkManaged && renderer->world()) {
            // One slot per chunk: defer the chunk's single slot until its
            // matching re-publish completes (or it ages out in
            // processPendingMeshes). Don't free immediately — for solid/water
            // the octree node is reused on edit (same NodeID), so republishing
            // the chunk updates the slot in place and consumes this entry.
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
            return;
        }

        // Legacy (append-based) main scene or brush scene: the chunk's mesh
        // is removed immediately from the target's chunk map + indirect
        // renderer. Brush meshes live in the brush IR (slotted removal); the
        // legacy main scene uses the append-based removal path.
        {
            std::lock_guard<std::recursive_mutex> lock(target.chunksMutex);
            auto it = target.chunks.find(nid);
            if (it != target.chunks.end()) {
                if (it->second.meshId != UINT32_MAX) {
                    if (target.chunkManaged)
                        target.indirect.removeMesh(it->second.meshId);
                    else
                        target.indirect.removeMeshSlotted(it->second.meshId);
                }
                target.chunks.erase(it);
            }
        }
        if (target.chunkManaged) {
            if (renderer->debugCubeRenderer) renderer->debugCubeRenderer->removeCubeForNode(nid);
            if (renderer->debugSDFRenderer) renderer->debugSDFRenderer->removeCubesForNode(nid);
        }
    };
    return { onAdded, onDeleted };
}

class MyApp : public VulkanApp, public IEventHandler {
public:
    Settings settings;
    SceneRenderer * sceneRenderer = nullptr;
    World * world = nullptr;
    std::shared_ptr<Brush3dWidget> brush3dWidget;
    // Shared brush entries edited by Brush3dWidget (owned by MyApp)
    Brush3dManager brushManager;
    // Cached sweep start position so applyBrushToScene uses the same pair as the preview
    glm::vec3 cachedSweepStart = glm::vec3(0.0f);
    static constexpr uint32_t QUERY_COUNT = 20; // 10 intervals × 2 timestamps each
    std::array<VkQueryPool, MAX_FRAMES_IN_FLIGHT> queryPools = {};
    bool queryPoolReady[MAX_FRAMES_IN_FLIGHT] = {};
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileShadow = 0.0f;
    float profileMainCull = 0.0f;
    float profileBrush = 0.0f;
    float profileDepthPrepass = 0.0f;
    float profileSky = 0.0f;
    float profileSolidDraw = 0.0f;
    float profileVegetationImpostor = 0.0f;
    float profileWater = 0.0f;
    float profilePostProcess = 0.0f;
    float profileImGui = 0.0f;
    float profileSolid360 = 0.0f;
    float profileBackface = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    float profileFps = 0.0f;
#ifdef DEBUG
    uint32_t vramWatchdogCounter_ = 0;
#endif
    UniformObject uboStatic = {};
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderTargetsWidget> renderTargetsWidget;
    std::shared_ptr<BillboardCreator> billboardCreator;
    std::shared_ptr<ImpostorService> impostorService;
    std::shared_ptr<ImpostorWidget> impostorWidget;
    std::shared_ptr<TextureMixerWidget> textureMixerWidget;
    // flag set by background thread when mixer widget is ready; main thread will add it safely
    bool mixerWidgetPendingAdd = false;
    std::shared_ptr<TextureViewer> textureViewer;
    std::shared_ptr<CameraWidget> cameraWidget;
    ControllerManager controllerManager;
    std::shared_ptr<ControllerParametersWidget> controllerParametersWidget;
    std::shared_ptr<GamepadWidget> gamepadWidget;
    std::shared_ptr<LightWidget> lightWidget;
    std::shared_ptr<VulkanResourcesManagerWidget> vulkanResourcesManagerWidget;
    std::shared_ptr<VegetationAtlasEditor> vegetationAtlasEditor;
    std::shared_ptr<WindWidget> windWidget;
    std::shared_ptr<MusicWidget> mp3Widget;
    std::shared_ptr<OctreeExplorerWidget> octreeExplorerWidget;
    std::shared_ptr<RadialMenu> radialMenu;
    std::unique_ptr<RadialMenuHandler> radialMenuHandler;
    WidgetManager widgetManager;
    FilePicker scenePicker_{"Scene File Picker", ".scene"};
    uint32_t loadedTextureLayers = 0;

    // Billboard editing / vegetation resources
    BillboardManager billboardManager;
    AtlasManager vegetationAtlasManager;
    TextureArrayManager vegetationTextureArrayManager = TextureArrayManager();

    // Global texture arrays and material manager (moved from SceneRenderer)
    TextureArrayManager textureArrayManager = TextureArrayManager();
    MaterialManager materialManager = MaterialManager();

    // Texture editing / UI helpers
    std::shared_ptr<TextureMixer> textureMixer;
    std::vector<MixerParameters> mixerParams;
    std::vector<MaterialProperties> materials;
    // Application-owned per-layer water parameters (initialized in setup)
    std::vector<WaterParams> waterParams;
    float mainTime = 0.0f;
    // Accumulated time driving the animated brush-space sphere trajectory
    float brushAnimTime = 0.0f;
    // Last frame delta, forwarded to postSubmit for the per-frame brush rebuild
    float lastFrameDelta = 0.0f;
    ShadowParams shadowParams;
    // When user clicks "Apply Brush" from ImGui we defer the heavy rebuild
    // until after the current frame is submitted to avoid waiting on fences
    // while the frame is being recorded (causes deadlock). Set by UI,
    // consumed in `postSubmit()`.
    bool brushRebuildPending = false;
    bool brushApplyToScenePending = false;
    bool generateMapPending = false;
    bool loadScenePending = false;
    std::string pendingLoadPath;
    size_t cubeCount = 0;

    // Camera and input
    Camera camera = Camera(glm::vec3(2673.0f, 125.0f, 2043.0f), Math::eulerToQuat(0.0f, 0.0f, 0.0f));
    Light light = Light(glm::vec3(-1.0f, -1.0f, -1.0f));
    EventManager eventManager;
    KeyboardPublisher keyboardPublisher;
    GamepadPublisher gamepadPublisher;
    NunchukPublisher nunchukPublisher;
    MousePublisher mousePublisher;
    bool sceneLoading = false;

    // One unified space-change handler per space: each owns both the
    // {onAdded, onDeleted} renderer lambdas and the dedup collector feeding
    // them (scene solid + scene liquid are the two "main" scene spaces).
    // Kept alive as members so the tessellation background thread can safely
    // dispatch them after setup() returns.
    std::thread sceneProcessThread; // tessellates chunks after octree is built

    // Pre-allocated descriptor pool+set rings to avoid per-frame create/destroy
    static constexpr uint32_t ASYNC_RING_SIZE = 3;
    struct PoolSetPair { VkDescriptorPool pool; VkDescriptorSet set; };
    PoolSetPair cachedBackfaceCompute[ASYNC_RING_SIZE]{};
    uint32_t ringBackfaceCompute = 0;


    Octree::OctreeNodeDataHandler brushSolidAddHandler;
    Octree::OctreeNodeDataHandler brushLiquidAddHandler;
    Octree::OctreeNodeDataHandler mainSolidAddHandler;
    Octree::OctreeNodeDataHandler mainLiquidAddHandler;
    
    Octree::OctreeNodeDataHandler brushSolidRemoveHandler;
    Octree::OctreeNodeDataHandler brushLiquidRemoveHandler;
    Octree::OctreeNodeDataHandler mainSolidRemoveHandler;
    Octree::OctreeNodeDataHandler mainLiquidRemoveHandler;

    UniqueChangeCollector mainSolidCollector;
    UniqueChangeCollector mainLiquidCollector;
    UniqueChangeCollector brushSolidCollector;
    UniqueChangeCollector brushLiquidCollector;
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
        Buffer compact{};                          // cull output: VkDrawIndexedIndirectCommand[]
        Buffer visible{};                          // cull output: draw count (uint32_t)
        uint32_t compactCapacity = 0;              // elements `compact` can hold
        VkDescriptorPool pool = VK_NULL_HANDLE;    // per-slot pool (maxSets=1)
        VkDescriptorSet waterDs = VK_NULL_HANDLE;  // water-depth set, rewritten per task
    };
    BackfaceSlot cachedBackfaceRing[ASYNC_RING_SIZE]{};
    uint32_t ringBackface = 0;
    ThreadPool asyncThreadPool{1}; // single worker for per-frame back-face pass

    // Persistent cubemap resources (used inline on main CB, no async race)
    Buffer cube360UBO{};
    Buffer cube360Compact{};
    Buffer cube360Visible{};
    Buffer cube360WaterCompact{};
    Buffer cube360WaterVisible{};
    VkDescriptorSet cube360GfxDs = VK_NULL_HANDLE;
    VkDescriptorSet cube360ComputeDs = VK_NULL_HANDLE;
    VkDescriptorSet cube360WaterComputeDs = VK_NULL_HANDLE;
    std::array<VkBuffer, 10> cube360ComputeBuffers = {};
    std::array<VkBuffer, 10> cube360WaterComputeBuffers = {};
    uint32_t cube360TexVersion = 0;

    ~MyApp() {}

    // setupTextures (defined out-of-line to avoid inline/member-definition issues)
    void setupTextures() {
        uint32_t layerCount = 32;

        textureArrayManager.allocate(layerCount, 1024, 1024, this);
        // Use shared TextureTriple defined in TextureArrayManager.hpp
        const std::vector<TextureTriple> textureTriples = {
            { "textures/Wall_Stone_010_basecolor.jpg", "textures/Wall_Stone_010_normal.jpg", "textures/Wall_Stone_010_height.jpg", "textures/Wall_Stone_010_roughness.jpg", "textures/Wall_Stone_010_ambientOcclusion.jpg" },
            { "textures/Ground_Dirt_007_basecolor.jpg", "textures/Ground_Dirt_007_normal.jpg", "textures/Ground_Dirt_007_height.jpg", "textures/Ground_Dirt_007_roughness.jpg", "textures/Ground_Dirt_007_ambientOcclusion.jpg" },
            { "textures/Dead_leaves_001_COLOR.jpg", "textures/Dead_leaves_001_NRM.jpg", "textures/Dead_leaves_001_DISP.jpg", "textures/Dead_leaves_001_SPEC.jpg", "textures/Dead_leaves_001_OCC.jpg" },
            { "textures/Grass_001_COLOR.jpg", "textures/Grass_001_NORM.jpg", "textures/Grass_001_DISP.jpg", "textures/Grass_001_ROUGH.jpg", "textures/Grass_001_OCC.jpg" },
            { "textures/Lava_005_COLOR.jpg", "textures/Lava_005_NORM.jpg", "textures/Lava_005_DISP.jpg", "textures/Lava_005_ROUGH.jpg", "textures/Lava_005_OCC.jpg" },
            { "textures/Metal_Pattern_008_basecolor.jpg", "textures/Metal_Pattern_008_normal.jpg", "textures/Metal_Pattern_008_height.jpg", "textures/Metal_Pattern_008_roughness.jpg", "textures/Metal_Pattern_008_ambientOcclusion.jpg" },
            { "textures/Linoleum_Floor_001_basecolor.jpg", "textures/Linoleum_Floor_001_normal.jpg", "textures/Linoleum_Floor_001_height.jpg", "textures/Linoleum_Floor_001_roughness.jpg", "textures/Linoleum_Floor_001_ambientOcclusion.jpg" },
            { "textures/Rough_rock_021_COLOR.jpg", "textures/Rough_rock_021_NRM.jpg", "textures/Rough_rock_021_DISP.jpg", "textures/Rough_rock_021_SPEC.jpg", "textures/Rough_rock_021_OCC.jpg" },
            { "textures/Sand_007_basecolor.jpg", "textures/Sand_007_normal.jpg", "textures/Sand_007_height.jpg", "textures/Sand_007_roughness.jpg", "textures/Sand_007_ambientOcclusion.jpg" },
            { "textures/Snow_001_COLOR.jpg", "textures/Snow_001_NORM.jpg", "textures/Snow_001_DISP.jpg", "textures/Snow_001_ROUGH.jpg", "textures/Snow_001_OCC.jpg" },
            { "textures/Sand_002_COLOR.jpg", "textures/Sand_002_NRM.jpg", "textures/Sand_002_DISP.jpg", "textures/Sand_002_SPEC.jpg", "textures/Sand_002_OCC.jpg" },
            { "textures/Bark_001_COLOR.jpg", "textures/Bark_001_NORM.jpg", "textures/Bark_001_DISP.jpg", "textures/Bark_001_ROUGH.jpg", "textures/Bark_001_OCC.jpg" },
            { "textures/Concrete_Blocks_013_basecolor.jpg", "textures/Concrete_Blocks_013_normal.jpg", "textures/Concrete_Blocks_013_height.jpg", "textures/Concrete_Blocks_013_roughness.jpg", "textures/Concrete_Blocks_013_ambientOcclusion.jpg" },
            { "textures/Asphalt_001_COLOR.jpg", "textures/Asphalt_001_NRM.jpg", "textures/Asphalt_001_DISP.jpg", "textures/Asphalt_001_SPEC.jpg", "textures/Asphalt_001_OCC.jpg" },
            { "textures/Stone_Floor_002_COLOR.jpg", "textures/Stone_Floor_002_NORM.jpg", "textures/Stone_Floor_002_DISP.jpg", "textures/Stone_Floor_002_SPEC.jpg", "textures/Stone_Floor_002_OCC.jpg" },
            { "textures/Canyon_Rock_001_COLOR.jpg", "textures/Canyon_Rock_001_NORM.jpg", "textures/Canyon_Rock_001_DISP.jpg", "textures/Canyon_Rock_001_ROUGH.jpg", "textures/Canyon_Rock_001_OCC.jpg" },
            { "textures/Sapphire_001_COLOR.jpg", "textures/Sapphire_001_NORM.jpg", "textures/Sapphire_001_DISP.jpg", "textures/Sapphire_001_ROUGH.jpg", "textures/Sapphire_001_OCC.jpg" },
            { "textures/Rough_rock_006_COLOR.jpg", "textures/Rough_rock_006_NRM.jpg", "textures/Rough_rock_006_DISP.jpg", "textures/Rough_rock_006_SPEC.jpg", "textures/Rough_rock_006_OCC.jpg" },
            { "textures/Crystal_Metal_001_COLOR.jpg", "textures/Crystal_Metal_001_NORM.jpg", "textures/Crystal_Metal_001_DISP.jpg", "textures/Crystal_Metal_001_ROUGH.jpg", "textures/Crystal_Metal_001_OCC.jpg" },
            { "textures/Sci-fi_Armor_001_basecolor.jpg", "textures/Sci-fi_Armor_001_normal.jpg", "textures/Sci-fi_Armor_001_height.jpg", "textures/Sci-fi_Armor_001_roughness.jpg", "textures/Sci-fi_Armor_001_ambientOcclusion.jpg" },
            { "textures/Greeble_Techno_002_basecolor.jpg", "textures/Greeble_Techno_002_normal.jpg", "textures/Greeble_Techno_002_height.jpg", "textures/Greeble_Techno_002_roughness.jpg", "textures/Greeble_Techno_002_ambientOcclusion.jpg" },

        };

        // Bulk load the triples directly using TextureTriple vector already defined above
        loadedTextureLayers = textureArrayManager.loadTriples(this, textureTriples);
        // Ensure mixer descriptor sets are updated with newly loaded arrays
        
        textureMixer = std::make_shared<TextureMixer>();
        textureMixer->init(this, &textureArrayManager);
        textureMixer->attachTextureArrayManager(&textureArrayManager);
        // Record into member so UI can display counts
        mixerParams.clear();
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 3u, 8u}); // grassMixSand
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 3u, 9u}); // grassMixSnow
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 3u}); // rockMixGrass
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 9u}); // rockMixSnow
        mixerParams.push_back(MixerParameters{loadedTextureLayers++, 7u, 8u}); // rockMixSand

        for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
            textureArrayManager.getImTexture(i, 0);
            textureArrayManager.getImTexture(i, 1);
            textureArrayManager.getImTexture(i, 2);
            textureArrayManager.getImTexture(i, 3);
            textureArrayManager.getImTexture(i, 4);
        }

        uint32_t editableLayer = (loadedTextureLayers < layerCount) ? loadedTextureLayers : 0u;
        (void)std::max(layerCount, std::max(loadedTextureLayers, 1u));



        // Trigger initial generation for configured mixers so UI previews show meaningful results
        // (Previously this was deferred to the user pressing "Generate" in the UI)
        std::cerr << "[TextureMixer] Running initial generation for configured mixers...\n";
        textureMixer->setEditableLayer(editableLayer);
        // Prime ImGui descriptors so the texture viewer shows immediately
        textureArrayManager.setLayerInitialized(editableLayer, true);
        textureArrayManager.getImTexture(editableLayer, 0);
        textureArrayManager.getImTexture(editableLayer, 1);
        textureArrayManager.getImTexture(editableLayer, 2);
        textureArrayManager.getImTexture(editableLayer, 3);
        textureArrayManager.getImTexture(editableLayer, 4);

        // Generate textures for all configured mixer entries (async submissions tracked by TextureMixer)
        textureMixer->generateInitialTextures(mixerParams);
        textureMixerWidget = std::make_shared<TextureMixerWidget>(textureMixer, mixerParams, "Texture Mixer");
        widgetManager.addWidget(textureMixerWidget);

        size_t materialCount = std::max<size_t>(static_cast<size_t>(loadedTextureLayers), static_cast<size_t>(loadedTextureLayers + 1));
        if (materialCount == 0) {
            materialCount = layerCount ? layerCount : 1u;
        }
        materials.assign(materialCount, MaterialProperties{});
          
        materials[0u].mappingMode = true;
        materials[0u].tessLevel = 5.0f;
        materials[0u].tessMinLevel = 2.0f;
        materials[0u].tessMaxLevel = 16.0f;
        materials[0u].tessHeightScale = 8.0f;
        materials[0u].triplanar = true;
        materials[0u].triplanarScaleU = 0.002f;
        materials[0u].triplanarScaleV = 0.002f;
        materials[0u].invertHeight = true;

        materials[5u].mappingMode = true;
        materials[5u].tessLevel = 5.0f;
        materials[5u].tessMinLevel = 2.0f;
        materials[5u].tessMaxLevel = 16.0f;
        materials[5u].tessHeightScale = 2.0f;
        materials[5u].triplanar = true;
        materials[5u].triplanarScaleU = 0.002f;
        materials[5u].triplanarScaleV = 0.002f;
        materials[5u].invertHeight = true;
        materials[5u].reflectionStrength = 0.8f;

        materials[6u].mappingMode = true;
        materials[6u].tessLevel = 1.0f;
        materials[6u].tessMinLevel = 1.0f;
        materials[6u].tessMaxLevel = 1.0f;
        materials[6u].tessHeightScale = 0.0f;
        materials[6u].triplanar = true;
        materials[6u].triplanarScaleU = 0.002f;
        materials[6u].triplanarScaleV = 0.002f;
        materials[6u].invertHeight = true;
        materials[6u].reflectionStrength = 1.0f;

        materials[7u].mappingMode = true;
        materials[7u].tessLevel = 2.0f;
        materials[7u].tessMinLevel = 2.0f;
        materials[7u].tessMaxLevel = 16.0f;
        materials[7u].tessHeightScale = 8.0f;
        materials[7u].triplanar = true;
        materials[7u].triplanarScaleU = 0.01f;
        materials[7u].triplanarScaleV = 0.01f;
        materials[7u].invertHeight = true;
        materials[7u].reflectionStrength = 0.2f;

        materials[12u].mappingMode = true;
        materials[12u].tessLevel = 6.0f;
        materials[12u].tessMinLevel = 2.0f;
        materials[12u].tessMaxLevel = 16.0f;
        materials[12u].tessHeightScale = 32.0f;
        materials[12u].triplanar = true;
        materials[12u].triplanarScaleU = 0.002f;
        materials[12u].triplanarScaleV = 0.002f;
        materials[12u].invertHeight = true;
        materials[12u].reflectionStrength = 0.3f;

        materials[14u].mappingMode = true;
        materials[14u].tessLevel = 5.0f;
        materials[14u].tessMinLevel = 2.0f;
        materials[14u].tessMaxLevel = 16.0f;
        materials[14u].tessHeightScale = 8.0f;
        materials[14u].triplanar = true;
        materials[14u].triplanarScaleU = 0.002f;
        materials[14u].triplanarScaleV = 0.002f;
        materials[14u].invertHeight = true;

        materials[15u].mappingMode = true;
        materials[15u].tessLevel = 5.0f;
        materials[15u].tessMinLevel = 2.0f;
        materials[15u].tessMaxLevel = 16.0f;
        materials[15u].tessHeightScale = 8.0f;
        materials[15u].triplanar = true;
        materials[15u].triplanarScaleU = 0.002f;
        materials[15u].triplanarScaleV = 0.002f;
        materials[15u].invertHeight = true;

        materials[16u].mappingMode = true;
        materials[16u].tessLevel = 5.0f;
        materials[16u].tessMinLevel = 2.0f;
        materials[16u].tessMaxLevel = 16.0f;
        materials[16u].tessHeightScale = 32.0f;
        materials[16u].triplanar = true;
        materials[16u].triplanarScaleU = 0.002f;
        materials[16u].triplanarScaleV = 0.002f;
        materials[16u].invertHeight = true;

        materials[18u].mappingMode = true;
        materials[18u].tessLevel = 5.0f;
        materials[18u].tessMinLevel = 2.0f;
        materials[18u].tessMaxLevel = 16.0f;
        materials[18u].tessHeightScale = 32.0f;
        materials[18u].triplanar = true;
        materials[18u].triplanarScaleU = 0.002f;
        materials[18u].triplanarScaleV = 0.002f;
        materials[18u].invertHeight = true;

        materials[20u].mappingMode = true;
        materials[20u].tessLevel = 5.0f;
        materials[20u].tessMinLevel = 2.0f;
        materials[20u].tessMaxLevel = 16.0f;
        materials[20u].tessHeightScale = 32.0f;
        materials[20u].triplanar = true;
        materials[20u].triplanarScaleU = 0.002f;
        materials[20u].triplanarScaleV = 0.002f;
        materials[20u].invertHeight = true;

        // Allocate GPU-side material storage via MaterialManager
        materialManager.allocate(materialCount, this);
        for (size_t i = 0; i < materialCount; ++i) materialManager.update(i, materials[i], this);

        // The descriptor set was bound to a dummy buffer at SceneRenderer::init() time
        // because materialManager hadn't been allocated yet (runs on a background thread).
        // Now that the real buffer exists, rebind descriptor set binding 5 so the GPU
        // reads from the actual materials SSBO.
        if (sceneRenderer) {
            sceneRenderer->updateTextureDescriptorSet(this, &textureArrayManager);
        }

    }

    void setup() override {
        sceneRenderer = new SceneRenderer();
        for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i)
            shadowParams.shadowMapSizes[i] = sceneRenderer->shadowMapper->getShadowMapSize(i);
        // Initialize application-owned water params with two default elements
        waterParams.push_back(WaterParams{}); // Add a third layer to demonstrate pagination in UI even without texture arrays
        {
            WaterParams wp = WaterParams();
            wp.noiseOctaves = 1;
            wp.waveScale = 8.0f;
            wp.deepColor = glm::vec3(0.0f, 0.1f, 0.0f);
            wp.causticColor = glm::vec3(0.5f, 1.0f, 0.0f);
            wp.shallowColor = glm::vec3(0.1f, 0.5f, 0.1f);
            wp.waterTint = 0.6f;
            wp.causticType = 1; // line-shaped caustics
            wp.causticIntensity = 0.2f;
            wp.causticScale = 0.01f;
            wp.causticDepthScale = 128.0f;
            wp.causticPower = 4.0f;
            wp.causticLineScale = 3.0f;
            wp.tessMinLevel = 2.0f;
            wp.tessMaxLevel = 16.0f;
            wp.reflectionStrength = 0.5f;
            wp.fresnelPower = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }
        {
            WaterParams wp = WaterParams();
            wp.enableRefraction = false;
            wp.noiseOctaves = 0;
            wp.waveScale = 0.0f;
            wp.noiseScale = 0.0f;
            wp.deepColor = glm::vec3(0.0f, 0.0f, 0.0f);
            wp.causticColor = glm::vec3(1.0f, 1.0f, 1.0f);
            wp.shallowColor = glm::vec3(1.0f, 1.0f, 1.0f);
            wp.waterTint = 1.0f;
            wp.causticType = 1; // line-shaped caustics
            wp.causticIntensity = 0.0f;
            wp.causticDepthScale = 128.0f;
            wp.causticPower = 4.0f;
            wp.causticLineScale = 3.0f;
            wp.bumpAmplitude = 0.0f;
            wp.blurRadius = 4.0f;
            wp.enableBlur = true;
            wp.reflectionStrength = 1.0f;
            wp.fresnelPower = 1.0f;
            wp.tessMinLevel = 1.0f;
            wp.tessMaxLevel = 1.0f;
            waterParams.push_back(wp); // Add a third layer to demonstrate pagination in UI even without texture arrays
        }


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

        octreeExplorerWidget = std::make_shared<OctreeExplorerWidget>(&world->scene(), &camera);
        widgetManager.addWidget(octreeExplorerWidget);
        world->createBrushScene();
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
        // main-scene space; the dedup collectors in front of them (fed to
        // Scene::loadScene/action and Octree::apply) replay final per-node
        // state into these handlers on the tessellation threads.
        Scene* sceneForChanges = &world->scene();
        float minSize = 30.0f;
        std::pair<Octree::OctreeNodeDataHandler,Octree::OctreeNodeDataHandler> mainOpaqueHandlers = build(
            sceneRenderer, 
            this, 
            sceneForChanges, 
            LAYER_OPAQUE, 
            minSize, 
            &sceneRenderer->mainSolidGenPool,
            
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->mainSolidChunks,
                sceneRenderer->mainSolidChunksMutex,
                sceneRenderer->mainSolidRenderer->getIndirectRenderer(), 
                sceneRenderer->pendingDeleteSolidSlots, 
                true
            }
        );
        mainSolidAddHandler = mainOpaqueHandlers.first;
        mainSolidRemoveHandler = mainOpaqueHandlers.second;

        std::pair<Octree::OctreeNodeDataHandler,Octree::OctreeNodeDataHandler> mainTransparentHandlers = build(
            sceneRenderer, 
            this, 
            sceneForChanges, 
            LAYER_TRANSPARENT, 
            minSize, 
            &sceneRenderer->mainWaterGenPool,
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->mainLiquidChunks,
                sceneRenderer->mainLiquidChunksMutex,
                sceneRenderer->mainLiquidRenderer->getIndirectRenderer(), 
                sceneRenderer->pendingDeleteWaterSlots, 
                true
            }
        );
        mainLiquidAddHandler = mainTransparentHandlers.first;
        mainLiquidRemoveHandler = mainTransparentHandlers.second;

        std::pair<Octree::OctreeNodeDataHandler,Octree::OctreeNodeDataHandler> brushOpaqueHandlers = build(
            sceneRenderer,
            this, 
            world->brushScene(), 
            LAYER_OPAQUE, 
            minSize, 
            &sceneRenderer->brushRenderer->solidGenPool,
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->brushRenderer->solidChunks,
                sceneRenderer->brushRenderer->solidChunksMutex,
                sceneRenderer->brushRenderer->getSolidIR(), 
                sceneRenderer->pendingDeleteSolidSlots, 
                false
            }
        );
        brushSolidAddHandler = brushOpaqueHandlers.first;
        brushSolidRemoveHandler = brushOpaqueHandlers.second;

        std::pair<Octree::OctreeNodeDataHandler,Octree::OctreeNodeDataHandler> brushTransparentHandlers = build(
            sceneRenderer,
            this, 
            world->brushScene(), 
            LAYER_TRANSPARENT, 
            minSize, 
            &sceneRenderer->brushRenderer->liquidGenPool,
            {
                sceneRenderer->pendingMeshQueue,
                sceneRenderer->pendingMeshMutex,
                sceneRenderer->brushRenderer->transparentChunks,
                sceneRenderer->brushRenderer->transparentChunksMutex,
                sceneRenderer->brushRenderer->getLiquidIR(), 
                sceneRenderer->pendingDeleteWaterSlots, 
                false
            }
        );
        brushLiquidAddHandler = brushTransparentHandlers.first;
        brushLiquidRemoveHandler = brushTransparentHandlers.second;



        // 5. Brush collectors are members; rebuildBrushScene feeds them via
        // apply and dispatches on the main thread.


        // Scene starts empty — use File > Generate Map to populate it.
        if (octreeExplorerWidget)

        // Init the VegetationRenderer before setupVegetationTextures so that
        // wind params UBO + descriptor set layout exist before captureAll calls
        // setImpostorData().  SceneRenderer::init() is called later after all
        // texture/material setup is complete.
        if (sceneRenderer && sceneRenderer->vegetationRenderer)
            sceneRenderer->vegetationRenderer->init(this);

        setupVegetationTextures();
        setupTextures();

        sceneRenderer->init(this, &textureArrayManager, &materialManager, waterParams);

        // Re-wire impostors now that VegetationRenderer::init() has stored the render pass.
        if (impostorService) impostorService->rewire();

        // Keep the vegetation array manager wired for editor/atlas updates.
        if (sceneRenderer->vegetationRenderer)
            sceneRenderer->vegetationRenderer->setTextureArrayManager(&vegetationTextureArrayManager, this);

        // Bind billboard array textures (sampler2DArray per channel) to the vegetation renderer.
        if (sceneRenderer->vegetationRenderer && billboardCreator) {
            sceneRenderer->vegetationRenderer->setBillboardArrayTextures(
                billboardCreator->getAlbedoArrayView(),
                billboardCreator->getNormalArrayView(),
                billboardCreator->getOpacityArrayView(),
                billboardCreator->getArraySampler(),
                this
            );
        }

        printf("[MyApp::setup] Created and initialized SceneRenderer\n");

        textureViewer = std::make_shared<TextureViewer>();
        textureViewer->init(&textureArrayManager, &materials);
        textureViewer->setOnMaterialChanged([this](size_t idx) {
            materialManager.update(idx, materials[idx], this);
        });

        skyWidget = std::make_shared<SkyWidget>(sceneRenderer->getSkySettings());
        // Create settings widget (was missing previously)
        settingsWidget = std::make_shared<SettingsWidget>(settings, &shadowParams);
        // Water UI uses the application-owned water params vector and updates GPU state explicitly.
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer->mainLiquidRenderer.get(), &waterParams);

        renderTargetsWidget = std::make_shared<RenderTargetsWidget>(
            this,
            sceneRenderer, sceneRenderer->mainSolidRenderer.get(), sceneRenderer->skyRenderer.get(),
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
        windWidget = std::make_shared<WindWidget>(sceneRenderer->vegetationRenderer.get());
        mp3Widget = std::make_shared<MusicWidget>();

        // Radial menu (input-agnostic overlay, not a Widget subclass)
        radialMenu = std::make_shared<RadialMenu>();
        radialMenuHandler = std::make_unique<RadialMenuHandler>(
            getWindow(), &eventManager, radialMenu.get(),
            &nunchukPublisher, &gamepadPublisher, &controllerManager,
            &brushManager, &textureArrayManager, &light);
        radialMenuHandler->setupPages();
  // Create octree explorer widget bound to loaded scene

 
        widgetManager.addWidget(textureViewer);
        widgetManager.addWidget(cameraWidget);
        widgetManager.addWidget(controllerParametersWidget);
        widgetManager.addWidget(gamepadWidget);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(lightWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderTargetsWidget);
        widgetManager.addWidget(vulkanResourcesManagerWidget);
        widgetManager.addWidget(vegetationAtlasEditor);
        widgetManager.addWidget(windWidget);
        widgetManager.addWidget(mp3Widget);
        widgetManager.addWidget(billboardCreator);
        widgetManager.addWidget(impostorWidget);

      
        // Subscribe event handlers
        eventManager.subscribe(&camera);  // Camera handles translate/rotate events
        eventManager.subscribe(this);     // MyApp handles close/fullscreen events

        // Each controller owns an independent page tree; subscribe them so
        // PageNavigationEvents (e.g. keyboard switching the mouse pages) reach
        // the right context.
        controllerManager.subscribeContexts(eventManager);

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
   
        // Create brush3dWidget after setupTextures() so loadedTextureLayers is set.
        brush3dWidget = std::make_shared<Brush3dWidget>(&textureArrayManager, loadedTextureLayers, brushManager, &eventManager);
        widgetManager.addWidget(brush3dWidget);
        rebuildBrushScene();

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
            }
            // Stall hook: when drawFrame detects a GPU ring hang (frame-slot
            // fence wait >0.5s or stale frame timeline), read every pool with
            // PARTIAL_BIT (no wait) — the pool of the stuck frame has start
            // timestamps written but its end timestamp missing, which names the
            // exact pass the GPU is stuck in. Also dump the submission ring.
            onFrameStall = [this](uint32_t) {
                static const char* intervalNames[10] = {
                    "shadow", "cull", "brush", "depth", "sky",
                    "solid", "veg", "water", "post", "imgui"
                };
                for (uint32_t f = 0; f < 3; ++f) {
                    if (queryPools[f] == VK_NULL_HANDLE) continue;
                    std::array<uint64_t, QUERY_COUNT> ts{};
                    if (vkGetQueryPoolResults(getDevice(), queryPools[f], 0, QUERY_COUNT,
                            sizeof(ts[0]) * ts.size(), ts.data(), sizeof(ts[0]),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_PARTIAL_BIT) != VK_SUCCESS)
                        continue;
                    std::cerr << "[stall] pool " << f << " timestamps:\n";
                    for (uint32_t i = 0; i < 10; ++i) {
                        const uint64_t start = ts[i * 2], end = ts[i * 2 + 1];
                        const bool haveStart = start != 0, haveEnd = end != 0;
                        std::cerr << "[stall]   " << intervalNames[i]
                                  << (haveEnd ? " DONE " : (haveStart ? " STUCK " : "  idle "))
                                  << " start=" << start << " end=" << end << "\n";
                    }
                }
            };
        }
        preAllocateAsyncDescriptorPools();
        // Try loading the default scene; fall back to procedural generation if it fails
        const std::string defaultScenePath = "scenes/default.scene";
        if (std::filesystem::exists(defaultScenePath)) {
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
    // Move scene-loading into its own method for clarity
    void setupScene();
    // Pre-allocate descriptor pool+set rings for async tasks
    void preAllocateAsyncDescriptorPools();
    // Rebuild the brush preview scene from Brush3dWidget entries
    void rebuildBrushScene();
    // Apply the selected brush SDF to the main scene's octree on the selected layer
    void applyBrushToScene();
    // When brush animation is enabled, advance the trajectory time and move the
    // selected brush entry along a circular orbit; the actual brush rebuild is
    // performed by rebuildBrushScene() afterwards.
    void updateBrushAnimation(float deltaTime);
    // Clear GPU meshes, reset octrees and regenerate via MainSceneLoader
    void generateMap();
    void action();
    // Clear GPU meshes, reset octrees, load from file and tessellate
    void loadSceneFromFile(const std::string& path);
    // Shared scene state reset (join thread, wait GPU, clear meshes/octrees/handlers)
    void resetSceneState();
    // Replay the deduplicated scene change events into the renderer handlers
    // (replaces UniqueOctreeChangeHandler::handleEvents).
    void dispatchSolidEvents();
    void dispatchLiquidEvents();

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
                                            world ? &world->scene().opaqueOctree : nullptr);
        } else {
            // Still poll nunchuk state so Home/A button edge detection works
            nunchukPublisher.update();
            // Poll gamepad left stick for radial menu input
            gamepadPublisher.pollLeftStick();
        }

        // Mouse: suppress when radial menu is visible or ImGui captures mouse
        bool mouseSuppressed = ImGui::GetIO().WantCaptureMouse || radialMenuVisible;
        mousePublisher.update(&eventManager, camera, deltaTime, &controllerManager,
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
        // Chunks closest to the camera are uploaded first. Drains both the
        // main scene and brush scene entries from the ONE shared queue.
        if (sceneRenderer && !isLoading) {
            std::deque<SceneRenderer::PendingMeshData> pendingBatch;
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
        // fixed per-frame cap. (Terrain/water/brush copies currently still go
        // through IndirectRenderer; this is the integration point for migrating
        // them onto UploadManager.)
        if (sceneRenderer)
            sceneRenderer->streamer.update(this);

        // Drain the CPU vegetation-generation queue so chunkBuffers is
        // populated before preRenderPass records read barriers.  Must
        // happen here because barriers cannot be emitted inside dynamic
        // rendering, and draw() runs inside beginPass/endPass.
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
            sceneRenderer->skyRenderer->update(this);
        }

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

    void preRenderPass(VkCommandBuffer &commandBuffer) override {

        uint32_t frameIdx = getCurrentFrame();

        // Per-frame cull buffers: each IndirectRenderer needs its own per-frame
        // compact/visibleCount buffer to avoid cross-frame overwrite races.
        // Must be set BEFORE prepareCull below so culls and drawPrepared use
        // the same per-frame compact/visibleCount slots (setCullFrame in
        // draw() would make every draw read a stale, never-culled slot).
        sceneRenderer->mainSolidRenderer->getIndirectRenderer().setCullFrame(frameIdx);
        sceneRenderer->brushRenderer->getSolidIR().setCullFrame(frameIdx);
        sceneRenderer->mainLiquidRenderer->getIndirectRenderer().setCullFrame(frameIdx);
        if (sceneRenderer->debugSDFRenderer) {
            sceneRenderer->debugSDFRenderer->setCullFrame(frameIdx);
            // The solid IndirectRenderer performs the SDF cube cull + compaction in
            // its OWN indirect.comp dispatch (folded into the terrain cull), so point
            // the SDF debug renderer at it to draw from its SDF output buffers.
            sceneRenderer->debugSDFRenderer->setIndirectRenderer(&sceneRenderer->mainSolidRenderer->getIndirectRenderer());
        }
        if (sceneRenderer->boundingBoxRenderer) {
            sceneRenderer->boundingBoxRenderer->setCullFrame(frameIdx);
            // Bounding-box frustum cull is folded into the solid IndirectRenderer's
            // indirect.comp dispatch, so draw from its bbox output buffers.
            sceneRenderer->boundingBoxRenderer->setIndirectRenderer(&sceneRenderer->mainSolidRenderer->getIndirectRenderer());
        }

        // Profiling: read previous frame's query results (with availability flag to
        // avoid even partial driver stalls), then reset for this frame.
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE) {
            if (queryPoolReady[frameIdx]) {
                auto msDiff = [&](uint64_t endTs, uint64_t startTs) -> float {
                    return static_cast<float>(endTs - startTs) * timestampPeriod * 1e-6f;
                };
                // Group A: indices 0-9 (shadow, cull, brush, depth prepass, sky)
                struct { uint64_t value; uint64_t availability; } tsA[10] = {};
                if (vkGetQueryPoolResults(getDevice(), queryPools[frameIdx], 0, 10,
                        sizeof(tsA), tsA, sizeof(tsA[0]),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS
                        && timestampPeriod > 0.0f) {
                    if (tsA[0].availability) profileShadow       = msDiff(tsA[1].value, tsA[0].value);
                    if (tsA[2].availability) profileMainCull     = msDiff(tsA[3].value, tsA[2].value);
                    if (tsA[4].availability) profileBrush        = msDiff(tsA[5].value, tsA[4].value);
                    if (tsA[6].availability) profileDepthPrepass = msDiff(tsA[7].value, tsA[6].value);
                    if (tsA[8].availability) profileSky          = msDiff(tsA[9].value, tsA[8].value);
                }
                // Group B: indices 10-19 (solid draw, veg impostor, water, postprocess, imgui)
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
            }
            // Throttled console dump of the previous frame's GPU passes so slow
            // frames are attributable from run.log without the ImGui panel.
            {
                static uint32_t profilePrintTick = 0;
                if ((++profilePrintTick & 0x1F) == 0) {
                    const float gpuTotal = profileShadow + profileMainCull + profileBrush +
                        profileDepthPrepass + profileSky + profileSolidDraw +
                        profileVegetationImpostor + profileWater + profilePostProcess +
                        profileImGui;
                    if (gpuTotal > 40.0f) {
                        std::cout << "[gpu] total=" << gpuTotal
                                  << " shadow=" << profileShadow
                                  << " cull=" << profileMainCull
                                  << " brush=" << profileBrush
                                  << " depth=" << profileDepthPrepass
                                  << " sky=" << profileSky
                                  << " solid=" << profileSolidDraw
                                  << " veg=" << profileVegetationImpostor
                                  << " water=" << profileWater
                                  << " post=" << profilePostProcess
                                  << " imgui=" << profileImGui
                                  << " fps=" << profileFps << std::endl;
                    }
                }
            }
            vkCmdResetQueryPool(commandBuffer, queryPools[frameIdx], 0, QUERY_COUNT);
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
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
        uboStatic.lightSpaceMatrix  = shadowParams.lightSpaceMatrix[0];
        uboStatic.lightSpaceMatrix1 = shadowParams.lightSpaceMatrix[1];
        uboStatic.lightSpaceMatrix2 = shadowParams.lightSpaceMatrix[2];
        // Encode debug/triplanar/tess parameters into the shared UBO
        uboStatic.debugParams = glm::vec4(static_cast<float>(settings.debugMode), settings.roughnessEnabled ? 1.0f : 0.0f, settings.aoEnabled ? 1.0f : 0.0f, 0.0f);
        uboStatic.triplanarSettings = glm::vec4(settings.triplanarThreshold, settings.triplanarExponent, 0.0f, 0.0f);
        uboStatic.tessParams = glm::vec4(
            settings.tessMinDistance,
            settings.tessMaxDistance,
            settings.tessellationFactor,
            0.0f
        );
        // passParams: x = isShadowPass, y = tessEnabled, z = nearPlane, w = farPlane
        uboStatic.passParams = glm::vec4(0.0f, settings.tessellationEnabled ? 1.0f : 0.0f, settings.nearPlane, settings.farPlane);
        // materialFlags.w = global normal-mapping toggle (shader checks ubo.materialFlags.w > 0.5)
        uboStatic.materialFlags.w = settings.normalMappingEnabled ? 1.0f : 0.0f;
        // shadowEffects.w = global shadow toggle (shader checks ubo.shadowEffects.w > 0.5)
        uboStatic.shadowEffects.w = settings.enableShadows ? 1.0f : 0.0f;

        // Brush params: brushTextureIndex, brushMode, brushHSV
        {
            float brushTexIdx = 0.0f;
            float brushMode = 0.0f;
            glm::vec3 brushHSV(0.0f, 0.5f, 0.5f);
            const BrushEntry* brushEntry = brushManager.getSelectedEntry();
            if (brushEntry) {
                brushTexIdx = static_cast<float>(brushEntry->materialIndex);
                brushMode = static_cast<float>(brushEntry->brushMode);
                brushHSV = brushEntry->hsv;
            }
            float brushTime = static_cast<float>(glfwGetTime());
            uboStatic.brushParams = glm::vec4(brushTexIdx, brushMode, 0.0f, brushTime);
            uboStatic.brushHSV = glm::vec4(brushHSV, 0.0f);
        }

        // Reset command buffer state tracker and wire it to all sub-renderers.
        // NOTE: backFaceRenderer and the water IndirectRenderer are deliberately
        // excluded by SceneRenderer::setCmdState — they are accessed by the
        // async back-face task on a separate thread and keeping cmdState=nullptr
        // for them avoids a data race on frameCmdState.
        sceneRenderer->frameCmdState.reset();
        sceneRenderer->setCmdState(&sceneRenderer->frameCmdState);

        // ── GPU culling: must run BEFORE shadow pass so drawPrepared has
        // current-frame compact/visibleCount buffers populated. ──
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 2);
        sceneRenderer->mainSolidRenderer->getIndirectRenderer().acquireBuffers(commandBuffer);
        // Vegetation cull is MERGED into the solid IndirectRenderer's single
        // indirect.comp dispatch: run vegetation prepareCull (consolidation +
        // per-frame buffer/metadata registration) BEFORE the solid prepareCull so
        // the merged dispatch knows about the veg outputs.
        if (sceneRenderer->vegetationRenderer && settings.vegetationEnabled) {
            sceneRenderer->vegetationRenderer->prepareCull(commandBuffer, viewProj);
        }
        // Fold SDF debug-cube culling into the solid IndirectRenderer's single
        // indirect.comp dispatch: register the cube AABBs BEFORE prepareCull so the
        // terrain cull writes the surviving SDF cubes into a dedicated SDF stream.
        // The cached per-chunk cubes are read internally by the renderer.
        if (settings.showSDFDebug && sceneRenderer && sceneRenderer->debugSDFRenderer) {
            sceneRenderer->debugSDFRenderer->registerToIndirect();
        }
        // Fold mesh bounding-box culling into the solid IndirectRenderer's single
        // indirect.comp dispatch: register the box AABBs BEFORE prepareCull so the
        // terrain cull writes the surviving boxes into a dedicated bbox stream. The
        // box list and the AABB list are built in lockstep (internal to the renderer)
        // so the cull's firstInstance (local index) matches the instance buffer.
        if (settings.showBoundingBoxes && sceneRenderer && sceneRenderer->boundingBoxRenderer) {
            sceneRenderer->boundingBoxRenderer->registerBoundingBoxesToIndirect();
        } else if (sceneRenderer && sceneRenderer->boundingBoxRenderer) {
            sceneRenderer->boundingBoxRenderer->clearBoundingBoxesToIndirect();
        }
        sceneRenderer->mainSolidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
        sceneRenderer->brushRenderer->getSolidIR().acquireBuffers(commandBuffer);
        sceneRenderer->brushRenderer->getSolidIR().prepareCull(commandBuffer, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
        // GPU frustum cull water meshes BEFORE the shadow pass so the water
        // shadow cascade reads the fresh (current-frame) LoD selection from the
        // shared visibleLods buffer. prepareCull acquires the water buffers
        // internally and must run outside a render pass.
        if (settings.waterEnabled && sceneRenderer->mainLiquidRenderer) {
            sceneRenderer->mainLiquidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
        }
        // Brush liquid (painted water) is drawn inside the water geometry pass
        // with the water pipeline, so it needs a current-frame cull of its own
        // (its compact/visibleCount buffers are otherwise stale from the last
        // prepareCull or uninitialized, which would draw garbage).
            sceneRenderer->brushRenderer->getLiquidIR().prepareCull(commandBuffer, viewProj, camera.getPosition(), settings.lodBias, settings.maxTargetLod);
        // Upload SDF debug-cube instance payload (frustum cull happens in the
        // solid IndirectRenderer's indirect.comp dispatch, folded into the terrain
        // cull). Must run outside the render pass, before the debug draw.
        if (settings.showSDFDebug && sceneRenderer && sceneRenderer->debugSDFRenderer) {
            // Refreshes the instance payload from the per-chunk cache internally.
            sceneRenderer->debugSDFRenderer->prepareCull(commandBuffer);
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 3);

        // Shadow pass: renders solid geometry into shadow map from light's perspective
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 0);
        if (sceneRenderer) {
            sceneRenderer->shadowMapper->render(this, commandBuffer, frameIdx, sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, settings.enableShadows, settings.renderSolid, settings.vegetationEnabled, settings.shadowTessellationEnabled, settings.lodBias, camera.getPosition(), settings.maxTargetLod);
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 1);

        // Upload UBO to GPU (VMA persistently mapped — no map/unmap needed)
        if (sceneRenderer) {
            memcpy(sceneRenderer->mainUniformBuffers[frameIdx].mappedData, &uboStatic, sizeof(UniformObject));
        } else {
            std::cerr << "[MyApp::preRenderPass] sceneRenderer is null, skipping UBO upload\n";
        }

        const bool waterEnabled = settings.waterEnabled;
        const bool vegetationEnabled = settings.vegetationEnabled;

        // Render sky + solids/vegetation into the solid offscreen framebuffer (one per frame)

        if (sceneRenderer->skyRenderer) {
            SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
            sceneRenderer->skyRenderer->renderOffscreen(this, commandBuffer, frameIdx,
                getMainDescriptorSet(), sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, viewProj, skyMode);
        }

        VkClearValue colorClear{};
        // Clear solid offscreen color to transparent so composite starts from empty scene
        colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkClearValue depthClear{};
        depthClear.depthStencil = {1.0f, 0};

            // Acquire vegetation instance/indirect buffers before
            // vkCmdBeginRendering (barriers illegal inside dynamic rendering).
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->recordReadBarriers(commandBuffer);
            }

        // Transition depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for Instance 1 below.
        // (The previous frame left it in SHADER_READ_ONLY after the water pass.)
        {
            VkImage solidDepthImg = sceneRenderer->mainSolidRenderer->getDepthImage(frameIdx);
            if (solidDepthImg != VK_NULL_HANDLE) {
                RendererUtils::transitionImageLayout(
                    commandBuffer, solidDepthImg,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
                setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            }
        }

        // ── Early brush pass: render brush to its own buffers before solid/water ──
        // Brush front depth (LESS test, writes depth), backface depth (GREATER test),
        // and brush color are all written here before solid geometry touches the
        // scene depth buffer. A later overlay pass renders brush with opacity on top
        // of solid/water using the scene depth buffer for occlusion culling.
        // Recording lives in BrushRenderer::recordEarlyPass.
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 4);
        if (sceneRenderer->brushRenderer) {
            sceneRenderer->brushRenderer->recordEarlyPass(
                this, commandBuffer, frameIdx,
                *sceneRenderer->mainSolidRenderer,
                getMainDescriptorSet());
        }
        if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 5);

        // ── Cubemap render on main CB (after brush pass, reads brush depth textures) ──
        const bool solid360PreviewActive = renderTargetsWidget && renderTargetsWidget->isVisible() && renderTargetsWidget->isSolid360Preview();
        const bool renderCubemap = (waterEnabled || solid360PreviewActive) && sceneRenderer && sceneRenderer->solid360Renderer;
        if (renderCubemap) {
            ensureCubemapResources();

            UniformObject ubo360 = uboStatic;
            ubo360.materialFlags.x = 1.0f; // skipEnvMap flag

            auto tCubemap = std::chrono::high_resolution_clock::now();
            this->sceneRenderer->solid360Renderer->render(
                this, commandBuffer,
                this->sceneRenderer->skyRenderer.get(), this->sceneRenderer->getSkySettings().mode,
                this->sceneRenderer->mainSolidRenderer.get(),
                cube360GfxDs,
                this->sceneRenderer->brushRenderer->getDepthDescriptorSet(frameIdx),
                cube360UBO, ubo360,
                settings.renderSolid, waterEnabled,
                cube360ComputeDs,
                cube360Compact.buffer,
                cube360Visible.buffer,
                cube360WaterComputeDs,
                cube360WaterCompact.buffer,
                cube360WaterVisible.buffer,
                frameIdx);
            this->profileSolid360 = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - tCubemap).count();
        }

        // ── Instance 1: Deferred depth pre-pass (no color attachment) ──
        // Solid + vegetation write depth; impostors use single-pass (depth+color in Instance 2).
        {
            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = sceneRenderer->mainSolidRenderer->getDepthView(frameIdx);
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue = depthClear;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {static_cast<uint32_t>(getWidth()), static_cast<uint32_t>(getHeight())};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 0;
            ri.pColorAttachments = nullptr;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(commandBuffer, &ri);

            {
                VkViewport vp{0.0f, 0.0f, (float)getWidth(), (float)getHeight(), 0.0f, 1.0f};
                vkCmdSetViewport(commandBuffer, 0, 1, &vp);
                VkRect2D sc{{0, 0}, {(uint32_t)getWidth(), (uint32_t)getHeight()}};
                vkCmdSetScissor(commandBuffer, 0, 1, &sc);
            }

            // Solid geometry depth
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 6);
            if (settings.renderSolid) {
                sceneRenderer->mainSolidRenderer->drawDepth(commandBuffer, this, getMainDescriptorSet());
            }

            // Vegetation depth (impostors render depth+color in Instance 2)
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->drawDepth(this, commandBuffer, viewProj, camera.getPosition());
            }

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 7);

            vkCmdEndRendering(commandBuffer);
        }

        // Note: no barrier needed between instances — vkCmdEndRendering makes depth
        // writes available; Instance 2's LOAD_OP_LOAD waits on them.

        // Transition color to COLOR_ATTACHMENT_OPTIMAL for Instance 2 below.
        {
            VkImage solidColorImg = sceneRenderer->mainSolidRenderer->getColorImage(frameIdx);
            if (solidColorImg != VK_NULL_HANDLE) {
                RendererUtils::transitionImageLayout(
                    commandBuffer, solidColorImg,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
                setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, 1);
            }
        }

        // ── Instance 2: Color pass (load depth from prepass, LESS compare for impostors) ──
        {
            VkRenderingAttachmentInfo colorAtt{};
            colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            colorAtt.imageView = sceneRenderer->mainSolidRenderer->getColorView(frameIdx);
            colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAtt.clearValue = colorClear;

            VkRenderingAttachmentInfo depthAtt{};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.imageView = sceneRenderer->mainSolidRenderer->getDepthView(frameIdx);
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAtt.clearValue = depthClear;

            VkRenderingInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.offset = {0, 0};
            ri.renderArea.extent = {static_cast<uint32_t>(getWidth()), static_cast<uint32_t>(getHeight())};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &colorAtt;
            ri.pDepthAttachment = &depthAtt;

            vkCmdBeginRendering(commandBuffer, &ri);

            {
                VkViewport vp{0.0f, 0.0f, (float)getWidth(), (float)getHeight(), 0.0f, 1.0f};
                vkCmdSetViewport(commandBuffer, 0, 1, &vp);
                VkRect2D sc{{0, 0}, {(uint32_t)getWidth(), (uint32_t)getHeight()}};
                vkCmdSetScissor(commandBuffer, 0, 1, &sc);
            }

            // Sky first (background, no depth write)
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 8);
            if (sceneRenderer->skyRenderer) {
                SkySettings::Mode skyMode = sceneRenderer->getSkySettings().mode;
                sceneRenderer->skyRenderer->render(this, commandBuffer, getMainDescriptorSet(),
                    sceneRenderer->mainUniformBuffers[frameIdx], uboStatic, viewProj, skyMode);
            }
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 9);

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 10);

            // Solid geometry color (LESS_OR_EQUAL, no depth write)
            if (settings.renderSolid) {
                VkDescriptorSet brushDepthSet = sceneRenderer->brushRenderer->getDepthDescriptorSet(frameIdx);
                sceneRenderer->mainSolidRenderer->drawColor(commandBuffer, this, getMainDescriptorSet(), brushDepthSet);
            }

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 11);

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 12);

            // Vegetation color + impostor color+depth (single-pass depth write)
            if (vegetationEnabled && sceneRenderer->vegetationRenderer) {
                sceneRenderer->vegetationRenderer->drawColor(this, commandBuffer, viewProj, camera.getPosition());
            }

            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 13);

            // Debug renders on top
            const bool showOctreeDebug = octreeExplorerWidget && octreeExplorerWidget->getShowDebugCubes();
            if (showOctreeDebug) {
                std::vector<DebugCubeRenderer::CubeWithColor> widgetCubes;
                if (octreeExplorerWidget->isVisible()) {
                    const auto& wc = octreeExplorerWidget->getExpandedCubes();
                    widgetCubes.reserve(wc.size());
                    for (const auto& c : wc)
                        widgetCubes.push_back({BoundingBox(c.cube.getMin(), c.cube.getMax()), c.color});
                }
                // renderOverlay merges the per-node cache internally (no getCubes).
                sceneRenderer->debugCubeRenderer->renderOverlay(this, commandBuffer, getMainDescriptorSet(), widgetCubes);
            }

            if (settings.showBoundingBoxes && sceneRenderer && sceneRenderer->boundingBoxRenderer) {
                // Boxes + AABBs were gathered and registered with the solid IR before
                // its prepareCull (folded into the terrain cull). Draw the surviving
                // boxes from the IR's bbox stream.
                sceneRenderer->boundingBoxRenderer->render(this, commandBuffer, getMainDescriptorSet());
            }

            if (settings.showSDFDebug && sceneRenderer && sceneRenderer->debugSDFRenderer) {
                sceneRenderer->debugSDFRenderer->render(this, commandBuffer, getMainDescriptorSet());
            }

            if (settings.renderSolid && settings.wireframeMode && sceneRenderer) {
                sceneRenderer->mainSolidRenderer->drawWireframeOverlay(commandBuffer, this, getMainDescriptorSet());
            }

            vkCmdEndRendering(commandBuffer);
        }

        // Transition color+depth to SHADER_READ_ONLY for water pass compositing.
        // The water pass's initializeGeomDepthFromSceneDepth expects the scene depth
        // in SHADER_READ_ONLY_OPTIMAL (it transitions to TRANSFER_SRC internally).
        {
            VkImage solidColorImg = sceneRenderer->mainSolidRenderer->getColorImage(frameIdx);
            VkImage solidDepthImg = sceneRenderer->mainSolidRenderer->getDepthImage(frameIdx);
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
                vkCmdPipelineBarrier2(commandBuffer, &dep);
            }
            if (solidColorImg != VK_NULL_HANDLE)
                setImageLayoutTracked(solidColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            if (solidDepthImg != VK_NULL_HANDLE)
                setImageLayoutTracked(solidDepthImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
        }

        // If water is disabled, clear its offscreen targets here (outside any active
        // dynamic rendering instance) so the post-process compositor won't sample
        // stale content.
        if (!waterEnabled && sceneRenderer && sceneRenderer->mainLiquidRenderer) {
            sceneRenderer->mainLiquidRenderer->clearRenderTargets(this, commandBuffer, frameIdx);
        }

        // Launch asynchronous recording+submit for independent offscreen passes
        // using a persistent thread pool to avoid per-frame std::thread creation overhead.
        VkSemaphore semBackFace = VK_NULL_HANDLE;
        std::future<void> asyncBackFaceFuture;

        // Back-face depth for water
        if (waterEnabled && sceneRenderer && sceneRenderer->backFaceRenderer) {
            asyncBackFaceFuture = asyncThreadPool.enqueue([this, viewProj, frameIdx, &semBackFace]() {
                VulkanApp* app = this;
                VkCommandBuffer cmd = app->allocatePrimaryCommandBuffer();
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
                    std::cerr << "[Async] vkBeginCommandBuffer failed for backFace" << std::endl;
                    app->freeCommandBuffer(cmd);
                    return;
                }
                // Reuse a ring of pre-allocated per-task resources (cull output buffers)
                // instead of creating host-visible buffers every frame. Slot safety:
                // see the cachedBackfaceRing comment above -- the previous submission
                // using this slot (task N) has completed before task N+ASYNC_RING_SIZE
                // runs, so reusing (and on growth, replacing) the buffers cannot race
                // with the GPU.
                IndirectRenderer &ind = this->sceneRenderer->mainLiquidRenderer->getIndirectRenderer();
                uint32_t numCmds = std::max({
                    static_cast<uint32_t>(ind.getMeshCount()),
                    static_cast<uint32_t>(ind.getMeshCapacity()),
                    1u
                });
                BackfaceSlot& slot = cachedBackfaceRing[ringBackface++ % ASYNC_RING_SIZE];

                // Lazily create the slot's buffers once. The compact buffer is only
                // recreated when the cull capacity grows; the old buffer's last
                // submission (this slot's previous task) has completed (see above),
                // so destroying it in place is safe and needs no deferred destroy.
                if (slot.compact.buffer == VK_NULL_HANDLE || slot.compactCapacity < numCmds) {
                    if (slot.compact.buffer != VK_NULL_HANDLE)
                        app->destroyBuffer(slot.compact); // slot's previous task completed
                    slot.compact = app->createBuffer(sizeof(VkDrawIndexedIndirectCommand) * numCmds,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    slot.compactCapacity = numCmds;
                }
                if (slot.visible.buffer == VK_NULL_HANDLE) {
                    slot.visible = app->createBuffer(sizeof(uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                }

                VkDevice dev = app->getDevice();
                auto lazyComputeSlot = [&](PoolSetPair* ring, uint32_t& idx, VkDescriptorSetLayout layout, const char* label) -> PoolSetPair& {
                    auto& s = ring[idx++ % ASYNC_RING_SIZE];
                    if (s.pool != VK_NULL_HANDLE) return s;
                    if (layout == VK_NULL_HANDLE) return s;
                    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
                    VkDescriptorPoolCreateInfo pci{};
                    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                    pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
                    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
                    vkCreateDescriptorPool(dev, &pci, nullptr, &s.pool);
                    app->resources.addDescriptorPool(s.pool, label);
                    VkDescriptorSetAllocateInfo ai{};
                    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    ai.descriptorPool = s.pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
                    vkAllocateDescriptorSets(dev, &ai, &s.set);
                    app->resources.addDescriptorSet(s.set, label);
                    return s;
                };
                VkDescriptorSet computeDs = VK_NULL_HANDLE;
                {
                    VkDescriptorSetLayout bfLayout = ind.getComputeDescriptorSetLayout();
                    auto& dsSlot = lazyComputeSlot(cachedBackfaceCompute, ringBackfaceCompute, bfLayout, "Lazy cachedBackfaceCompute");
                    computeDs = dsSlot.set;
                }

                // Update descriptor set with buffers: inCmds, outCmds, bounds,
                // visibleCount, visibleLods (binding 4 = persistent scratch).
                if (computeDs != VK_NULL_HANDLE) {
                    DescriptorWriter(dev)
                        .writeBuffer(computeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getIndirectBuffer().buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     slot.compact.buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getBoundsBuffer().buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     slot.visible.buffer, 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVisibleLodsScratchBuffer(), 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                        .writeBuffer(computeDs, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     ind.getVegDummyBuffer(), 0, VK_WHOLE_SIZE)
                        .flush();
                }

                // Run cull into per-task buffers - only when compute pipeline is ready (meshes loaded)
                if (computeDs != VK_NULL_HANDLE) {
                    ind.prepareCullWithDescriptor(cmd, viewProj, computeDs, slot.compact.buffer, slot.visible.buffer,
                                                  camera.getPosition(), settings.lodBias, settings.maxTargetLod);
                }

                // Water-depth descriptor set for THIS task: pre-allocated per ring slot
                // and rewritten each frame before submission. Reuse is safe because the
                // slot's previous submission, which bound this set on the GPU, has
                // completed before we rewrite it (see the cachedBackfaceRing comment).
                // A dedicated set per slot keeps the async back-face pass from sharing
                // the per-frame set with the main command buffer (which would require
                // UPDATE_AFTER_BIND and trip GPU-assisted validation's descriptor-count
                // check).
                VkDescriptorSet asyncWaterDs = VK_NULL_HANDLE;
                {
                    VkImageView bfBack = (this->sceneRenderer->backFaceRenderer) ? this->sceneRenderer->backFaceRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
                    VkImageView bfCube = (this->sceneRenderer->solid360Renderer) ? this->sceneRenderer->solid360Renderer->getSolid360View() : VK_NULL_HANDLE;
                    VkDescriptorSetLayout wdsLayout = this->sceneRenderer->mainLiquidRenderer->getWaterDepthDescriptorSetLayout();
                    if (wdsLayout != VK_NULL_HANDLE && bfBack != VK_NULL_HANDLE && bfCube != VK_NULL_HANDLE) {
                        if (slot.pool == VK_NULL_HANDLE) {
                            // Per-slot pool (maxSets=1) so the set is allocated once and
                            // rewritten every task; flags mirror the renderer's async
                            // pool (the water-depth layout has no UPDATE_AFTER_BIND
                            // bindings, so none is required here).
                            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 };
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
                            VkImageView bfColor = this->sceneRenderer->mainSolidRenderer->getColorView(frameIdx);
                            VkImageView bfDepth = this->sceneRenderer->mainSolidRenderer->getDepthView(frameIdx);
                            VkImageView bfSky   = (this->sceneRenderer->skyRenderer) ? this->sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
                            this->sceneRenderer->mainLiquidRenderer->updateSceneTexturesBinding(this, slot.waterDs, bfColor, bfDepth, frameIdx, bfSky, bfBack, bfCube);
                            asyncWaterDs = slot.waterDs;
                        }
                    }
                }

                // Patch binding #3 to the dummy depth to avoid the back-face
                // pass reading-from the same image it writes-to as depth
                // attachment (SYNC-HAZARD READ_AFTER_WRITE).
                if (asyncWaterDs != VK_NULL_HANDLE) {
                    WaterBackFaceRenderer* bfr = this->sceneRenderer->backFaceRenderer.get();
                    if (bfr && bfr->getDummyDepthView() != VK_NULL_HANDLE) {
                        bfr->patchBinding3(asyncWaterDs, bfr->getDummyDepthView());
                    }
                }

                // Render back-face pass using this slot's (ring-reused) compact/visible
                // buffers so draws consume the cull results
                auto tBackface = std::chrono::high_resolution_clock::now();
                this->sceneRenderer->backFaceRenderer->render(app, cmd, frameIdx,
                                            ind,
                                            this->sceneRenderer->mainLiquidRenderer->getWaterGeometryPipelineLayout(),
                                            app->getMainDescriptorSet(),
                                            asyncWaterDs,
                                            (computeDs != VK_NULL_HANDLE) ? slot.compact.buffer : VK_NULL_HANDLE,
                                            (computeDs != VK_NULL_HANDLE) ? slot.visible.buffer : VK_NULL_HANDLE);

                this->profileBackface = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - tBackface).count();
                // Submit. The ring slot's buffers/set are NOT defer-destroyed: they are
                // reused ASYNC_RING_SIZE tasks later, by which time this submission has
                // completed (guaranteed by the frame-fence chain described on
                // cachedBackfaceRing). Use submitCommandBufferAsyncToQueue on the
                // graphics queue so the completion semaphore (semBackFace) is
                // registered in m_extraWaitSemaphores and drawFrame waits on it before
                // the main command buffer reads the back-face depth (e.g. in the water
                // tessellation evaluation shader). A plain submitCommandBufferAsync
                // would signal the semaphore but never register it, leaving the
                // cross-command-buffer write->read dependency unsynchronized.
                app->submitCommandBufferAsyncToQueue(cmd, app->getGraphicsQueue(), &semBackFace);
            });
        }

        // Wait for the async back-face task to complete before water pass so no two
        // threads call vkCmdBindDescriptorSets with the same descriptor set concurrently
        // and to ensure semBackFace is signaled before waterPass uses it. get() rethrows
        // task exceptions (wait() would swallow them, silently leaving semBackFace
        // unsignaled and dropping the back-face pass for the frame).
        if (asyncBackFaceFuture.valid()) {
            try {
                asyncBackFaceFuture.get();
            } catch (const std::exception &e) {
                std::cerr << "[Async] back-face task failed, skipping back-face pass this frame: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[Async] back-face task failed with unknown error, skipping back-face pass this frame" << std::endl;
            }
        }

        // Run water geometry pass offscreen and bind scene textures for post-process
        if (waterEnabled) {
            // Water frustum cull already ran before the shadow pass; re-assert
            // buffer visibility for the water geometry draw (no re-cull needed —
            // the shadow cascade cull uses separate buffers and did not disturb
            // the main water compact buffer).
            sceneRenderer->mainLiquidRenderer->getIndirectRenderer().acquireBuffers(commandBuffer);
            // Use 360° solid+sky reflection instead of the sky-only equirect view
            VkImageView skyView = (sceneRenderer && sceneRenderer->skyRenderer) ? sceneRenderer->skyRenderer->getSkyView(frameIdx) : VK_NULL_HANDLE;
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPools[frameIdx], 14);
            sceneRenderer->mainLiquidRenderer->renderPass(this, commandBuffer, frameIdx, settings.waterWireframeMode,
                mainTime, skyView);
            if (profilingEnabled && queryPools[frameIdx] != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPools[frameIdx], 15);

            // Transition water geometry depth to SRO for the PostProcess compositor
            VkImage wgdImg = sceneRenderer->mainLiquidRenderer->getWaterGeomDepthImage(frameIdx);
            if (wgdImg != VK_NULL_HANDLE) {
                recordTransitionImageLayoutLayer(commandBuffer, wgdImg,
                    VK_FORMAT_D32_SFLOAT,
                    sceneRenderer->mainLiquidRenderer->getWaterGeomDepthLayout(frameIdx),
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    1, 0, 1);
                sceneRenderer->mainLiquidRenderer->setWaterGeomDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        profileCpuRecord = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuRecordT0).count();
    }

    void renderImGui() override {
        static char sceneFolderBuf[512] = "scenes/default.scene";

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Generate Map")) generateMapPending = true;
                if (ImGui::MenuItem("Action")) action();
                ImGui::Separator();
                if (ImGui::MenuItem("Save Scene...")) scenePicker_.open(sceneFolderBuf, true,  "Mode: Save");
                if (ImGui::MenuItem("Load Scene...")) scenePicker_.open(sceneFolderBuf, false, "Mode: Load");
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
                    if (world) world->scene().save(sceneFolderBuf, &settings);
                } else {
                    pendingLoadPath = sceneFolderBuf;
                    loadScenePending = true;
                }
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

                // Opaque (solid)
                size_t opaqueLoaded = sceneRenderer->mainSolidRenderer->getIndirectRenderer().getMeshCount();
                uint32_t opaqueVisible = sceneRenderer->mainSolidRenderer->getIndirectRenderer().readVisibleCount(this);
                ImGui::Text("Opaque - Loaded (GPU): %zu  Visible (GPU cull): %u", opaqueLoaded, opaqueVisible);
                size_t opaqueTracked = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;
                ImGui::Text("Opaque Models Tracked: %zu", opaqueTracked);

                // Transparent / water
                size_t transparentLoaded = sceneRenderer && sceneRenderer->mainLiquidRenderer ? sceneRenderer->mainLiquidRenderer->getIndirectRenderer().getMeshCount() : 0;
                uint32_t transparentVisible = sceneRenderer && sceneRenderer->mainLiquidRenderer ? sceneRenderer->mainLiquidRenderer->getIndirectRenderer().readVisibleCount(this) : 0;
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
                    float gpuTotal = profileShadow + profileMainCull + profileBrush +
                                     profileDepthPrepass + profileSky + profileSolidDraw +
                                     profileVegetationImpostor + profileWater +
                                     profilePostProcess + profileImGui;
                    ImGui::Text("Shadow:        %.2f", profileShadow);
                    ImGui::Text("GPU Cull:      %.2f", profileMainCull);
                    ImGui::Text("Brush:         %.2f", profileBrush);
                    ImGui::Text("Depth Prepass: %.2f", profileDepthPrepass);
                    ImGui::Text("Sky:           %.2f", profileSky);
                    ImGui::Text("Solid Draw:    %.2f", profileSolidDraw);
                    ImGui::Text("Veg Impostor:  %.2f", profileVegetationImpostor);
                    ImGui::Text("Water:         %.2f", profileWater);
                    ImGui::Text("PostProcess:   %.2f", profilePostProcess);
                    ImGui::Text("ImGui:         %.2f", profileImGui);
                    ImGui::Text("--- GPU Total:  %.2f ---", gpuTotal);
                    ImGui::Separator();
                    ImGui::Text("--- CPU Timing (ms) ---");
                    ImGui::Text("Solid360*:     %.2f", profileSolid360);
                    ImGui::Text("Backface*:     %.2f", profileBackface);
                    ImGui::Text("* = CPU-timed (async)");
                    ImGui::Separator();
                    ImGui::Text("FPS:           %.1f", profileFps);
                    ImGui::Text("Update:        %.2f", profileCpuUpdate);
                    ImGui::Text("Record:        %.2f", profileCpuRecord);
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

        cubeCount = sceneRenderer ? sceneRenderer->getRegisteredModelCount() : 0;

        // Update per-frame widget state (avoid storing VulkanApp* inside widgets)
        if (renderTargetsWidget) renderTargetsWidget->setFrameInfo(getCurrentFrame(), getWidth(), getHeight());
        if (vulkanResourcesManagerWidget) vulkanResourcesManagerWidget->updateWithApp(this);

        // Render radial menu (behind all widgets)
        if (radialMenu) {
            radialMenu->Update();
            radialMenu->Draw();
        }

        // Render all widgets
        widgetManager.renderAll();
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
            VkImageView brushBackFaceDepthView = sceneRenderer->brushRenderer ? sceneRenderer->brushRenderer->getBackFaceDepthView(frameIdx) : VK_NULL_HANDLE;
            VkImageView waterGeomDepthView = VK_NULL_HANDLE;
            if (sceneRenderer->mainLiquidRenderer) {
                waterGeomDepthView = sceneRenderer->mainLiquidRenderer->getWaterGeomDepthView(frameIdx);
            }
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
                sceneRenderer->mainSolidRenderer->getColorView(frameIdx),
                sceneRenderer->mainSolidRenderer->getDepthView(frameIdx),
                sceneRenderer->mainLiquidRenderer->getWaterDepthView(frameIdx),
                brushColorView,
                brushDepthView,
                brushBackFaceDepthView,
                waterGeomDepthView,
                brushAlpha,
                brushMode,
                viewProj,
                invViewProj,
                glm::vec3(uboStatic.viewPos),
                frameIdx,
                skyViewPP);
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
            ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
            if (profilingEnabled && qp != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 19);
        }
    }

    void clean() override {
    // Ensure all GPU work is finished before tearing down any Vulkan resources.
    // Belt-and-suspenders: VulkanApp::cleanup() already calls deviceWaitIdle()
    // before clean(), but draining the upload queues and stopping thread pools
    // may have left work in flight.
    deviceWaitIdle();

    // Join tessellation thread before tearing down Vulkan resources.
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

        // Free all ImGui descriptor sets owned by the widget while ImGui is still
        // alive. clean() is called before cleanupImGui(), so this is safe.
        // Without this, ~RenderTargetsWidget() (called from ~MyApp() after Vulkan
        // teardown) would try to free descriptors from a destroyed pool.
        if (renderTargetsWidget) {
            renderTargetsWidget->invalidateImGuiDescriptors();
        }

        if (impostorService) {
            impostorService->invalidateImGuiDescriptors();
            impostorService->cleanup();
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
        for (auto& slot : cachedBackfaceCompute) slot = {};
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
        if (sceneProcessThread.joinable()) sceneProcessThread.join();
        asyncThreadPool.stop();
        if (world) world->stopPools();
        if (sceneRenderer) sceneRenderer->stopGenPools();
    }

    void onSwapchainResized(uint32_t width, uint32_t height) override {
        if (sceneRenderer) {
            sceneRenderer->onSwapchainResized(this, width, height);
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
        if (billboardCreator) {
            billboardCreator->invalidateImGuiDescriptors();
        }
        textureArrayManager.invalidateImGuiDescriptors();
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
        if (auto rebuildEvent = std::dynamic_pointer_cast<RebuildBrushEvent>(event)) {
            // Defer heavy rebuild to postSubmit() to avoid interfering with
            // command buffer recording and GPU fences.
            brushRebuildPending = true;
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
                brushRebuildPending = true;
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
    // Ensure persistent cubemap rendering resources are allocated
    void ensureCubemapResources();

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
    try {
        MyApp app;
        app.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

// Implementation: setup scene
void MyApp::setupScene() {
    // Scene objects, background thread, and brush3dWidget are now set up
    // directly in setup() so the CPU-heavy scene load can run in parallel
    // with texture loading. This stub is kept for call-site compatibility.
}

// Implementation: setup vegetation textures
void MyApp::setupVegetationTextures() {
    // Allocate 3-layer texture arrays for vegetation (foliage, grass, wild)
    vegetationTextureArrayManager.allocate(3, 512, 512, this);
    vegetationAtlasEditor = std::make_shared<VegetationAtlasEditor>(&vegetationTextureArrayManager, &vegetationAtlasManager);
    auto billboardSvc = std::make_shared<BillboardService>();
    billboardCreator = std::make_shared<BillboardCreator>(&billboardManager, &vegetationAtlasManager, &vegetationTextureArrayManager, billboardSvc);
    // Provide VulkanApp to the creator so it can initialize GPU-backed preview textures
    billboardCreator->setVulkanApp(this);
    impostorService = std::make_shared<ImpostorService>();
    // Must be set before init(): the capture pipeline shares the renderer's
    // set=2 wind params descriptor set/layout instead of duplicating them.
    impostorService->setVegetationRenderer(sceneRenderer->vegetationRenderer.get());
    impostorService->init(this);
    impostorWidget = std::make_shared<ImpostorWidget>(impostorService);
    impostorWidget->setVegetationRenderer(sceneRenderer->vegetationRenderer.get());
    
    // Load the vegetation atlas textures (albedo, normal, opacity) into the texture array
    std::vector<TextureTriple> vegTriples = {
        { "textures/vegetation/foliage_color.jpg", "textures/vegetation/foliage_normal.jpg", "textures/vegetation/foliage_opacity.jpg" },
        { "textures/vegetation/grass_color.jpg",   "textures/vegetation/grass_normal.jpg",   "textures/vegetation/grass_opacity.jpg" },
        { "textures/vegetation/wild_color.jpg",    "textures/vegetation/wild_normal.jpg",    "textures/vegetation/wild_opacity.jpg" }
    };
    size_t loaded = vegetationTextureArrayManager.loadTriples(this, vegTriples);
    std::cerr << "[MyApp::setupVegetationTextures] Loaded " << loaded << " vegetation texture layers" << std::endl;

    // Auto-detect atlas tiles from opacity maps and populate AtlasManager for each texture
    const char* opacityPaths[3] = { "textures/vegetation/foliage_opacity.jpg", "textures/vegetation/grass_opacity.jpg", "textures/vegetation/wild_opacity.jpg" };
    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        try {
            vegetationAtlasManager.clear(atlasIndex);
            int added = vegetationAtlasManager.autoDetectTiles(atlasIndex, opacityPaths[atlasIndex]);
            std::cerr << "[MyApp::setupVegetationTextures] Atlas " << atlasIndex << ": auto-detected " << added << " tiles" << std::endl;
        } catch (...) {
            std::cerr << "[MyApp::setupVegetationTextures] Atlas " << atlasIndex << ": autoDetectTiles failed" << std::endl;
        }
    }

    // Initialize the editor with 3 vegetation billboards.
    // Each billboard uses only one atlas (its respective texture), and includes
    // all available tiles from that atlas as layers with uniform horizontal offsets.
    billboardManager.clear();

    for (int atlasIndex = 0; atlasIndex < 3; ++atlasIndex) {
        const std::string name = "Vegetation Billboard " + std::to_string(atlasIndex + 1);
        const size_t bidx = billboardManager.createBillboard(name);

        Billboard* billboard = billboardManager.getBillboard(bidx);
        if (billboard) {
            billboard->width = 1.0f;
            billboard->height = 1.5f;
        }

        const size_t tileCount = vegetationAtlasManager.getTileCount(atlasIndex);
        if (tileCount == 0) continue;

        // Repeat tiles until there are at least minSlots to fill the full width.
        const size_t minSlots   = 5;
        const size_t repeats    = (minSlots + tileCount - 1) / tileCount;
        const size_t totalSlots = repeats * tileCount;

        for (size_t slot = 0; slot < totalSlots; ++slot) {
            const size_t tileIndex = slot % tileCount;
            const AtlasTile* tile  = vegetationAtlasManager.getTile(atlasIndex, static_cast<int>(tileIndex));

            BillboardLayer layer;
            layer.atlasIndex = atlasIndex;
            layer.tileIndex  = static_cast<int>(tileIndex);
            // Distribute slots uniformly left-to-right in normalised [-1,+1] space.
            // Centre of slot s of n: (2s+1)/n − 1
            layer.offsetX = (2.0f * static_cast<float>(slot) + 1.0f)
                            / static_cast<float>(totalSlots) - 1.0f;
            layer.offsetY = 0.0f;
            // Scale each tile to exactly fill its 1/totalSlots slot of billboard width.
            // In compositeLayer: visible_width = 2 * scaleX * tile->scaleX, desired = 2/totalSlots.
            layer.scaleX = (tile && tile->scaleX > 1e-6f)
                           ? 1.0f / (tile->scaleX * static_cast<float>(totalSlots))
                           : 1.0f;
            // Scale each tile to fill the full billboard height (100%).
            // denomY = scaleY * tile->scaleY; setting scaleY = 1/tile->scaleY → denomY = 1.0,
            // so the full [-1,+1] output range maps to the full tile height.
            layer.scaleY = (tile && tile->scaleY > 1e-6f)
                           ? 1.0f / tile->scaleY
                           : 1.0f;
            layer.rotation = 0.0f;
            layer.opacity  = 1.0f;
            layer.renderOrder = static_cast<int>(slot);

            billboardManager.addLayer(bidx, layer);
        }
    }

    if (billboardCreator) {
        // Bake authoring billboards into dedicated per-billboard GPU textures.
        billboardCreator->bakeAllBillboards();
    }

    // Wire freshly baked billboard array textures to VegetationRenderer.
    // The setup() call to setBillboardArrayTextures happens before baking,
    // so the views were VK_NULL_HANDLE. Re-wire now with valid views.
    if (sceneRenderer && sceneRenderer->vegetationRenderer && billboardCreator) {
        sceneRenderer->vegetationRenderer->setBillboardArrayTextures(
            billboardCreator->getAlbedoArrayView(),
            billboardCreator->getNormalArrayView(),
            billboardCreator->getOpacityArrayView(),
            billboardCreator->getArraySampler(),
            this
        );
    }

    // Notify ImpostorService about the freshly baked texture arrays.
    if (impostorService && billboardCreator) {
        impostorService->setSource(
            billboardCreator->getAlbedoArrayView(),
            billboardCreator->getNormalArrayView(),
            billboardCreator->getOpacityArrayView(),
            billboardCreator->getArraySampler(),
            static_cast<int>(billboardManager.getBillboardCount()));
    }
}

// Implementation: pre-allocate descriptor pool+set rings for async tasks
void MyApp::preAllocateAsyncDescriptorPools() {
    VkDevice dev = getDevice();

    auto allocateComputeRing = [&](PoolSetPair* ring, VkDescriptorSetLayout dsLayout, const char* label) {
        if (dsLayout == VK_NULL_HANDLE) return;
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        for (uint32_t i = 0; i < ASYNC_RING_SIZE; ++i) {
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(dev, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
                std::cerr << "[Async] Failed to pre-allocate " << label << " pool " << i << "\n";
                ring[i] = {};
                continue;
            }
            { std::string s = std::string(label) + " ring #" + std::to_string(i); resources.addDescriptorPool(pool, s.c_str()); }
            VkDescriptorSet set;
            VkDescriptorSetAllocateInfo ainfo{};
            ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ainfo.descriptorPool = pool;
            ainfo.descriptorSetCount = 1;
            ainfo.pSetLayouts = &dsLayout;
            if (vkAllocateDescriptorSets(dev, &ainfo, &set) != VK_SUCCESS) {
                resources.removeDescriptorPool(pool);
                vkDestroyDescriptorPool(dev, pool, nullptr);
                std::cerr << "[Async] Failed to pre-allocate " << label << " set " << i << "\n";
                ring[i] = {};
                continue;
            }
            { std::string s = std::string(label) + " DS #" + std::to_string(i); resources.addDescriptorSet(set, s.c_str()); }
            ring[i] = {pool, set};
        }
    };

    // Backface compute (same layout as water compute)
    if (sceneRenderer && sceneRenderer->mainLiquidRenderer) {
        auto& waterInd = sceneRenderer->mainLiquidRenderer->getIndirectRenderer();
        allocateComputeRing(cachedBackfaceCompute, waterInd.getComputeDescriptorSetLayout(), "cachedBackfaceCompute");
    }
}

// Shared SDF creation: populates fn2 (current) + optionally fn1 (sweep start),
// wraps in SweepSignedDistanceFunction when sweepMode is on, then calls callback.
template<typename Fn>
static void forEachBrushSDF(const BrushEntry& entry, const Transformation& model,
                            const glm::vec3& sweepStart, float minSize,
                            const char* logPrefix, Fn&& callback) {
    switch (entry.sdfType) {
        case 0: { // Sphere
            SphereDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                SphereDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<SphereDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 1: { // Box
            BoxDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                BoxDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<BoxDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 2: { // Capsule
            CapsuleDistanceFunction fn2(entry.capsuleA, entry.capsuleB, entry.capsuleRadius, model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                CapsuleDistanceFunction fn1(entry.capsuleA, entry.capsuleB, entry.capsuleRadius, prevModel, minSize);
                SweepSignedDistanceFunction<CapsuleDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 3: { // Octahedron
            OctahedronDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                OctahedronDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<OctahedronDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 4: { // Pyramid
            PyramidDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                PyramidDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<PyramidDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 5: { // Torus
            TorusDistanceFunction fn2(entry.torusRadii, model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                TorusDistanceFunction fn1(entry.torusRadii, prevModel, minSize);
                SweepSignedDistanceFunction<TorusDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 6: { // Cone
            ConeDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                ConeDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<ConeDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 7: { // Cylinder
            CylinderDistanceFunction fn2(model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                CylinderDistanceFunction fn1(prevModel, minSize);
                SweepSignedDistanceFunction<CylinderDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 8: { // Tapered Cylinder
            TaperedCylinderDistanceFunction fn2(entry.taperedCylinderRadii.x, entry.taperedCylinderRadii.y, model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                TaperedCylinderDistanceFunction fn1(entry.taperedCylinderRadii.x, entry.taperedCylinderRadii.y, prevModel, minSize);
                SweepSignedDistanceFunction<TaperedCylinderDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
            break;
        }
        case 9: { // Tapered Capsule
            TaperedCapsuleDistanceFunction fn2(entry.capsuleA, entry.capsuleB,
                entry.taperedCapsuleRadii.x, entry.taperedCapsuleRadii.y, model, minSize);
            if (entry.sweepMode) {
                Transformation prevModel(entry.scale, sweepStart, entry.rot);
                TaperedCapsuleDistanceFunction fn1(entry.capsuleA, entry.capsuleB,
                    entry.taperedCapsuleRadii.x, entry.taperedCapsuleRadii.y, prevModel, minSize);
                SweepSignedDistanceFunction<TaperedCapsuleDistanceFunction> sweepFn(fn1, fn2, model, minSize);
                callback(sweepFn);
            } else { callback(fn2); }
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
                                 const Transformation& model, const TexturePainter& brush,
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
                octree.apply(op, effect, model, brush, minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 1: {
                PerlinCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency, entry.effectThreshold,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, model, brush, minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 2: {
                SineDistortDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectFrequency, glm::vec3(0), model, minSize);
                octree.apply(op, effect, model, brush, minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            case 3: {
                VoronoiCarveDistanceEffect effect(func,
                    entry.effectAmplitude, entry.effectCellSize,
                    glm::vec3(0), entry.effectBrightness, entry.effectContrast, model, minSize);
                octree.apply(op, effect, model, brush, minSize, simplifier, updateHandler, deleteHandler);
                break;
            }
            default:
                octree.apply(op, func, model, brush, minSize, simplifier, updateHandler, deleteHandler);
                break;
        }
    } else {
        octree.apply(op, func, model, brush, minSize, simplifier, updateHandler, deleteHandler);
    }
}

// Implementation: rebuild the brush scene from Brush3dWidget entries
void MyApp::rebuildBrushScene() {
    if (!world || !world->brushScene() || !sceneRenderer || !brush3dWidget) return;
    if (getenv("SKIP_BRUSH")) {
        std::cerr << "[MyApp::rebuildBrushScene] SKIPPED (SKIP_BRUSH set)" << std::endl;
        return;
    }

    // No device-wide stall here. The brush flow uses the stable-slot indirect
    // pipeline: clearBrushMeshes() frees old slots, handleEvents() queues geometry,
    // and processPendingMeshes() commits each mesh independently via
    // addMeshSlotted() + uploadSlot() — no global rebuild required.

    // Process only the currently-selected brush entry from the manager
    const BrushEntry* selectedEntry = brushManager.getSelectedEntry();
    size_t selCount = selectedEntry ? 1 : 0;

    // Capture the start position for this frame's sweep (before any updates)
    if (selectedEntry && selectedEntry->sweepMode) {
        cachedSweepStart = selectedEntry->previousTranslate;
    }
    std::cerr << "[MyApp::rebuildBrushScene] Rebuilding with " << selCount << " selected entries" << std::endl;

    // 1. Stage existing brush meshes for smooth transition (don't clear until new ones are ready)
    sceneRenderer->brushRenderer->stageOldChunks();

    // 2. Reset the brush octrees (clears spatial data without change events)
    world->brushScene()->getOpaqueOctree().reset();
    world->brushScene()->transparentOctree.reset();

    if (!selectedEntry) {
        // Nothing to add — free staged old slots immediately.
        std::deque<SceneRenderer::PendingMeshData> pendingBatch;
        sceneRenderer->drainPendingMeshes(pendingBatch, 16);
        sceneRenderer->processPendingMeshes(this, camera.getPosition(), pendingBatch);
        return;
    }


    // 3. The brush {onAdded, onDeleted} renderer lambdas (built in setup()
    // with world->brushScene()) route geometry to the dedicated brush queue +
    // chunk maps. The octree invokes change handlers on its own worker
    // threads, so the renderer lambdas must NOT run during traversal — collect
    // here and dispatch() on this (main) thread below.


    // angle=0.95 (cos≈18°): normals within 18° → flat surface → full distance tolerance.
    // distance=0.2: flat patches may have up to 20% cube-size SDF error (curved gets 10%).
    Simplifier simplifier(0.95f, 0.2f, true);
    // 4. Process the selected brush entry only
    const auto& entry = *selectedEntry;
        // Select the target octree and handler based on targetLayer
        Octree& octree = (entry.targetLayer == 0)
            ? world->brushScene()->getOpaqueOctree()
            : world->brushScene()->transparentOctree;
        Octree::OctreeNodeDataHandler& updateHandler = (entry.targetLayer == 0)
            ? brushSolidCollector.updateHandler
            : brushLiquidCollector.updateHandler;
        Octree::OctreeNodeDataHandler& deleteHandler = (entry.targetLayer == 0)
            ? brushSolidCollector.deleteHandler
            : brushLiquidCollector.deleteHandler;

        Transformation model(entry.scale, entry.translate, entry.rot);
        SimpleBrush brush(entry.materialIndex, entry.hsv);

        // Create the base SDF primitive (stack-allocated, octree copies during add)
        // sdfType: 0=Sphere,1=Box,2=Capsule,3=Octahedron,4=Pyramid,5=Torus,6=Cone,7=Cylinder
        // We use a lambda to avoid massive switch duplication for add vs del with optional effects
        auto applyEntry = [&](SignedDistanceFunction& wrappedFunc) {
            AddSignedDistanceOperation brushOp;
            applyBrushWithEffect(entry, wrappedFunc, octree, brushOp, model, brush, simplifier, updateHandler, deleteHandler);
        };

        forEachBrushSDF(entry, model, cachedSweepStart, entry.minSize, "[rebuildBrushScene]", applyEntry);
    // 5. Flush queued change events on the MAIN thread (triggers mesh
    // creation via the SceneRenderer brush handlers).
    brushSolidCollector.dispatch(brushSolidAddHandler, brushSolidRemoveHandler);
    brushLiquidCollector.dispatch(brushLiquidAddHandler, brushLiquidRemoveHandler);

    // 6. Process all brush meshes IMMEDIATELY (synchronous, not deferred to
    // the next frame's update()). The brush scene is small — this avoids the
    // 1-frame delay where old chunks are removed and new ones are not yet
    // uploaded, eliminating the progressive "chunk by chunk" visual update.
    // Old staged slots are freed BEFORE new slots are allocated so the
    // 128-slot brush pool is never exhausted by stale old entries.
    std::deque<SceneRenderer::PendingMeshData> pendingBatch;
    sceneRenderer->drainPendingMeshes(pendingBatch, 16);
    sceneRenderer->processPendingMeshes(this, camera.getPosition(), pendingBatch);

    // Advance previousTranslate for next frame's sweep (frame-by-frame trail)
    if (selectedEntry && selectedEntry->sweepMode) {
        BrushEntry* mutableEntry = brushManager.getSelectedEntry();
        if (mutableEntry) {
            mutableEntry->previousTranslate = mutableEntry->translate;
        }
    }
}

void MyApp::applyBrushToScene() {
    if (!world || !sceneRenderer) return;

    const BrushEntry* selectedEntry = brushManager.getSelectedEntry();
    if (!selectedEntry) return;

    const auto& entry = *selectedEntry;

    // Select brush operation based on brushMode
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

    // Select target octree and handler based on targetLayer
    Octree& octree = (entry.targetLayer == 0)
        ? world->scene().opaqueOctree
        : world->scene().transparentOctree;

    Octree::OctreeNodeDataHandler& updateHandler = (entry.targetLayer == 0)
        ? mainSolidCollector.updateHandler
        : mainLiquidCollector.updateHandler;
    Octree::OctreeNodeDataHandler& deleteHandler = (entry.targetLayer == 0)
        ? mainSolidCollector.deleteHandler
        : mainLiquidCollector.deleteHandler;

    // cachedSweepStart was already set by rebuildBrushScene — use the same pair
    Transformation model(entry.scale, entry.translate, entry.rot);
    SimpleBrush brush(entry.materialIndex, entry.hsv);

    Simplifier simplifier(0.95f, 0.2f, true);
    auto applyEntry = [&](SignedDistanceFunction& wrappedFunc) {
        applyBrushWithEffect(entry, wrappedFunc, octree, brushOp, model, brush, simplifier, updateHandler, deleteHandler);
    };

    // Use cachedSweepStart from rebuild's START (before it advanced previousTranslate)
    forEachBrushSDF(entry, model, cachedSweepStart, entry.minSize, "[applyBrushToScene]", applyEntry);

    // Flush queued change events to trigger mesh creation
    mainSolidCollector.dispatch(mainSolidAddHandler, mainSolidRemoveHandler);
    mainLiquidCollector.dispatch(mainLiquidAddHandler, mainLiquidRemoveHandler);

    // Mark indirect buffers dirty so the mesh changes are visible
    sceneRenderer->mainSolidRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->mainSolidRenderer->getIndirectRenderer().rebuild(this);
    sceneRenderer->mainLiquidRenderer->getIndirectRenderer().setDirty(true);
    sceneRenderer->mainLiquidRenderer->getIndirectRenderer().rebuild(this);

    // Update previousTranslate for the next sweep apply
    if (entry.sweepMode) {
        BrushEntry* mutableEntry = brushManager.getSelectedEntry();
        if (mutableEntry) {
            mutableEntry->previousTranslate = mutableEntry->translate;
        }
    }
}

// Advance the brush-animation clock and move the *selected* brush entry along a
// circular trajectory layered on top of the existing triangle-strip ring
// (MainSceneLoader). Only the entry's translate is changed — its shape, size,
// texture, mode and layer are left intact, so the animation composes with the
// user's brush definition. The entry retains the last animated position when
// animation is disabled, allowing manual editing from there.
void MyApp::updateBrushAnimation(float deltaTime) {
    BrushEntry* entry = brushManager.getSelectedEntry();
    if (!entry) return;

    // Freeze the clock while disabled so re-enabling continues from the same
    // orbit phase (the entry keeps its last animated translate).
    brushAnimTime += deltaTime;

    // Don't override brush position while AIM subpage is active
    const ControllerPage* subpage = controllerManager.wiimoteContext.activeSubpage();
    if (subpage && subpage->control == PageControl::AIM) return;

    // Ring parameters (defined in MainSceneLoader): centered at origin, height
    // 800, outer radius = worldScale (1500). The brush orbits that ring.
    constexpr float ringHeight = 800.0f;
    constexpr float ringRadius = 1500.0f;   // worldScale * unitOuter(1.0)
    constexpr float orbitSpeed = 0.5f;      // radians / second

    float angle = brushAnimTime * orbitSpeed;
    entry->translate = glm::vec3(ringRadius * std::cos(angle), ringHeight,
                                 ringRadius * std::sin(angle));
}

// Ensure pending texture generation requests are flushed after a frame is submitted
// so array-layer transitions happen outside of active draw command buffers.

void MyApp::action() {
    // Join any previous background tessellation thread
    if (sceneProcessThread.joinable()) sceneProcessThread.join();

    // Wait for the GPU to finish all in-flight work before clearing GPU resources
    deviceWaitIdle();

    MainSceneLoader loader;
    world->scene().action(loader,
        mainSolidCollector.updateHandler, mainSolidCollector.deleteHandler,
        mainLiquidCollector.updateHandler, mainLiquidCollector.deleteHandler
    );
    std::cout << "[MyApp::action] Octree construction complete\n";

    // Tessellate chunks in a background thread. Solid and water are handled on
    // separate threads so both layers tessellate truly in parallel (water no
    // longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { dispatchSolidEvents(); });
        std::thread waterThread([this]() { dispatchLiquidEvents(); });
        solidThread.join();
        waterThread.join();

        std::cout << "[MyApp::action] Scene chunk tessellation complete\n";
    });
}

void MyApp::resetSceneState() {
    if (sceneProcessThread.joinable()) sceneProcessThread.join();
    deviceWaitIdle();
    processPendingCommandBuffers();

    if (sceneRenderer) {
        sceneRenderer->removeAllRegisteredMeshes();
        sceneRenderer->removeAllTransparentMeshes();
        world->chunkManager().removeAll();
        if (sceneRenderer->debugCubeRenderer) sceneRenderer->debugCubeRenderer->clearCubes();
        if (sceneRenderer->debugSDFRenderer) sceneRenderer->debugSDFRenderer->clearCubes();
        if (sceneRenderer->vegetationRenderer) {
            sceneRenderer->vegetationRenderer->clearAllInstances();
        }
    }

    world->scene().opaqueOctree.reset();
    world->scene().transparentOctree.reset();

    mainSolidCollector.clear();
    mainLiquidCollector.clear();
}

void MyApp::dispatchSolidEvents() {
    mainSolidCollector.dispatch(mainSolidAddHandler, mainSolidRemoveHandler);
}

void MyApp::dispatchLiquidEvents() {
    mainLiquidCollector.dispatch(mainLiquidAddHandler, mainLiquidRemoveHandler);
}

void MyApp::generateMap() {
    resetSceneState();

    // Build the octree (CPU only, no tessellation)
    MainSceneLoader loader;
    world->scene().loadScene(loader,
        mainSolidCollector.updateHandler, mainSolidCollector.deleteHandler,
        mainLiquidCollector.updateHandler, mainLiquidCollector.deleteHandler
    );
    std::cout << "[MyApp::generateMap] Octree construction complete\n";

    // Tessellate chunks in a background thread. Solid and water are handled on
    // separate threads so both layers tessellate truly in parallel (water no
    // longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { dispatchSolidEvents(); });
        std::thread waterThread([this]() { dispatchLiquidEvents(); });
        solidThread.join();
        waterThread.join();
        std::cout << "[MyApp::generateMap] Scene chunk tessellation complete\n";
    });
}

void MyApp::loadSceneFromFile(const std::string& path) {
    resetSceneState();

    world->scene().load(path,
        mainSolidCollector.updateHandler, mainSolidCollector.deleteHandler,
        mainLiquidCollector.updateHandler, mainLiquidCollector.deleteHandler,
        &settings);
    std::cout << "[MyApp::loadSceneFromFile] Octree loaded from '" << path << "'\n";

    // Solid and water tessellate on separate threads so both layers progress
    // truly in parallel (water no longer waits for solid to finish).
    sceneProcessThread = std::thread([this]() {
        std::thread solidThread([this]() { dispatchSolidEvents(); });
        std::thread waterThread([this]() { dispatchLiquidEvents(); });
        solidThread.join();
        waterThread.join();
        std::cout << "[MyApp::loadSceneFromFile] Scene tessellation complete\n";
    });
}
void MyApp::ensureCubemapResources() {
    VkDevice dev = getDevice();

    // Always write dummy cubemap to cube360GfxDs binding #11,
    // even if the DS was already allocated (e.g. before this fix was compiled).
    if (cube360GfxDs != VK_NULL_HANDLE && sceneRenderer && sceneRenderer->solid360Renderer) {
        VkImageView dummyView = sceneRenderer->solid360Renderer->getDummyCubeView();
        VkSampler cubeSamp = sceneRenderer->solid360Renderer->getSolid360Sampler();
        if (dummyView != VK_NULL_HANDLE && cubeSamp != VK_NULL_HANDLE) {
            DescriptorWriter(dev)
                .writeImage(cube360GfxDs, 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            cubeSamp, dummyView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                .flush();
        }
    }

    // 1. UBO buffer
    if (cube360UBO.buffer == VK_NULL_HANDLE) {
        cube360UBO = createBuffer(sizeof(UniformObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    // Helper to destroy and recreate a buffer if its size is insufficient
    auto ensureBufferSize = [&](Buffer& buf, VkDeviceSize needed,
                                VkBufferUsageFlags usage, const char* label) {
        if (buf.buffer != VK_NULL_HANDLE) {
            VkMemoryRequirements reqs;
            vkGetBufferMemoryRequirements(dev, buf.buffer, &reqs);
            if (reqs.size >= needed) return; // already large enough
            // The old buffer may still be referenced by command buffers that are
            // currently in flight (the cube360 culling pass runs every frame).
            // Destroying it synchronously here is the VUID-vkDestroyBuffer-buffer-00922
            // crash; defer the destruction until the GPU is idle instead.
            Buffer old = buf;
            deferDestroyUntilAllPending([old, this]() mutable {
                destroyBuffer(old);
            });
            buf = Buffer{};
        }
        buf = createBuffer(needed, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    // 2. Solid culling buffers (reallocate if mesh count grew)
    // NOTE: In slotted mode the indirect commands are pre-allocated to
    // meshCapacity (e.g. 1024); getMeshCount() returns only active meshes (0
    // before the first scene load).  The compact buffer must be sized to the
    // full slot pool so that drawPreparedWithBuffers' maxCount is valid.
    IndirectRenderer &solidInd = sceneRenderer->mainSolidRenderer->getIndirectRenderer();
    uint32_t solidCmds = std::max({
        static_cast<uint32_t>(solidInd.getMeshCount()),
        static_cast<uint32_t>(solidInd.getMeshCapacity()),
        1u
    });
    VkDeviceSize compactSize = sizeof(VkDrawIndexedIndirectCommand) * solidCmds;
    ensureBufferSize(cube360Compact, compactSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        "cube360Compact");
    ensureBufferSize(cube360Visible, std::max(sizeof(uint32_t), VkDeviceSize(4)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        "cube360Visible");

    // 3. Water culling buffers (reallocate if mesh count grew)
    IndirectRenderer &waterInd = sceneRenderer->mainLiquidRenderer->getIndirectRenderer();
    uint32_t waterCmds = std::max({
        static_cast<uint32_t>(waterInd.getMeshCount()),
        static_cast<uint32_t>(waterInd.getMeshCapacity()),
        1u
    });
    VkDeviceSize waterCompactSize = sizeof(VkDrawIndexedIndirectCommand) * waterCmds;
    ensureBufferSize(cube360WaterCompact, waterCompactSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        "cube360WaterCompact");
    ensureBufferSize(cube360WaterVisible, std::max(sizeof(uint32_t), VkDeviceSize(4)),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        "cube360WaterVisible");

    // 4. Graphics descriptor set (mirrors main DS but uses cube360UBO)
    if (cube360GfxDs == VK_NULL_HANDLE) {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 };
        VkDescriptorPoolSize ps2{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 11 };
        VkDescriptorPoolSize ps3{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolSize poolSizes[] = {ps, ps2, ps3};

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

        VkDescriptorPool gfxPool;
        if (vkCreateDescriptorPool(dev, &poolInfo, nullptr, &gfxPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create cubemap GFX descriptor pool");
        resources.addDescriptorPool(gfxPool, "cubemap gfx pool");

        VkDescriptorSetLayout gfxLayout = getDescriptorSetLayout();
        VkDescriptorSetAllocateInfo ainfo{};
        ainfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ainfo.descriptorPool = gfxPool;
        ainfo.descriptorSetCount = 1;
        ainfo.pSetLayouts = &gfxLayout;
        if (vkAllocateDescriptorSets(dev, &ainfo, &cube360GfxDs) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate cubemap GFX descriptor set");
        resources.addDescriptorSet(cube360GfxDs, "cubemap gfx DS");

        // Write descriptor set bindings using DescriptorWriter
        {
            DescriptorWriter gfxWriter(dev);
            gfxWriter.writeBuffer(cube360GfxDs, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  cube360UBO.buffer, 0, sizeof(UniformObject));

            auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
                if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
                gfxWriter.writeImage(cube360GfxDs, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     sampler, view, layout);
            };

            if (textureArrayManager.albedoSampler != VK_NULL_HANDLE) {
                addImg(1, textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(2, textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(3, textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(12, textureArrayManager.roughnessSampler, textureArrayManager.roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                addImg(13, textureArrayManager.aoSampler, textureArrayManager.aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            addImg(4, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(8, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(9, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            if (sceneRenderer->solid360Renderer) {
                VkImageView dummyCubeView = sceneRenderer->solid360Renderer->getDummyCubeView();
                VkSampler cubeSampler = sceneRenderer->solid360Renderer->getSolid360Sampler();
                if (dummyCubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE)
                    addImg(11, cubeSampler, dummyCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }

            if (sceneRenderer->materialsBuffer.buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      sceneRenderer->materialsBuffer.buffer, 0, VK_WHOLE_SIZE);
            if (sceneRenderer->waterParamsBuffer_.buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      sceneRenderer->waterParamsBuffer_.buffer, 0, VK_WHOLE_SIZE);
            {
                Buffer skyBuf = sceneRenderer->skyRenderer->getSkyUniformBuffer();
                if (skyBuf.buffer != VK_NULL_HANDLE)
                    gfxWriter.writeBuffer(cube360GfxDs, 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                          skyBuf.buffer, 0, sizeof(SkyUniform));
            }
            if (sceneRenderer->mainLiquidRenderer->getWaterRenderUBO().buffer != VK_NULL_HANDLE)
                gfxWriter.writeBuffer(cube360GfxDs, 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      sceneRenderer->mainLiquidRenderer->getWaterRenderUBO().buffer, 0, sizeof(WaterRenderUBO));

            gfxWriter.flush();
        }
        cube360TexVersion = textureArrayManager.getVersion();
    }

    // Refresh texture bindings on cube360GfxDs if texture arrays were re-allocated
    // (e.g. after TextureMixer generates new layers). Without this, the cubemap
    // capture would sample stale/deleted image views, producing wrong reflections.
    if (cube360GfxDs != VK_NULL_HANDLE && cube360TexVersion != textureArrayManager.getVersion()) {
        cube360TexVersion = textureArrayManager.getVersion();
        DescriptorWriter texWriter(dev);
        auto addImg = [&](uint32_t binding, VkSampler sampler, VkImageView view, VkImageLayout layout) {
            if (view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) return;
            texWriter.writeImage(cube360GfxDs, binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 sampler, view, layout);
        };
        if (textureArrayManager.albedoSampler != VK_NULL_HANDLE) {
            addImg(1, textureArrayManager.albedoSampler, textureArrayManager.albedoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(2, textureArrayManager.normalSampler, textureArrayManager.normalArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(3, textureArrayManager.bumpSampler, textureArrayManager.bumpArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(12, textureArrayManager.roughnessSampler, textureArrayManager.roughnessArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(13, textureArrayManager.aoSampler, textureArrayManager.aoArray.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (sceneRenderer && sceneRenderer->shadowMapper) {
            addImg(4, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(8, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(1), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            addImg(9, sceneRenderer->shadowMapper->getShadowMapSampler(), sceneRenderer->shadowMapper->getShadowMapView(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (sceneRenderer->solid360Renderer) {
            VkImageView dummyCubeView = sceneRenderer->solid360Renderer->getDummyCubeView();
            VkSampler cubeSampler = sceneRenderer->solid360Renderer->getSolid360Sampler();
            if (dummyCubeView != VK_NULL_HANDLE && cubeSampler != VK_NULL_HANDLE)
                addImg(11, cubeSampler, dummyCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        texWriter.flush();
    }

    // 5. Solid compute descriptor set — allocate lazily, refresh buffer bindings only when buffers change
    {
        VkDescriptorSetLayout dsLayout = solidInd.getComputeDescriptorSetLayout();
        if (dsLayout != VK_NULL_HANDLE && cube360ComputeDs == VK_NULL_HANDLE) {
            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
            pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(dev, &pci, nullptr, &pool) == VK_SUCCESS) {
                resources.addDescriptorPool(pool, "cubemap compute pool");
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsLayout;
                if (vkAllocateDescriptorSets(dev, &ai, &cube360ComputeDs) == VK_SUCCESS)
                    resources.addDescriptorSet(cube360ComputeDs, "cubemap compute DS");
            }
        }
        if (cube360ComputeDs != VK_NULL_HANDLE) {
            std::array<VkBuffer, 10> bufs = {
                solidInd.getIndirectBuffer().buffer,
                cube360Compact.buffer,
                solidInd.getBoundsBuffer().buffer,
                cube360Visible.buffer,
                solidInd.getVisibleLodsScratchBuffer(),
                solidInd.getVegDummyBuffer(), // 5: veg impostor cmds (unused here)
                solidInd.getVegDummyBuffer(), // 6: veg impostor count
                solidInd.getVegDummyBuffer(), // 7: veg billboard cmds
                solidInd.getVegDummyBuffer(), // 8: veg billboard count
                solidInd.getVegDummyBuffer(), // 9: veg chunk info
            };
            if (bufs != cube360ComputeBuffers) {
                cube360ComputeBuffers = bufs;
                DescriptorWriter(dev)
                    .writeBuffer(cube360ComputeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[0], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[1], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[2], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[3], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[4], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[5], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[6], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[7], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[8], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360ComputeDs, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[9], 0, VK_WHOLE_SIZE)
                    .flush();
            }
        }
    }

    // 6. Water compute descriptor set — allocate lazily, refresh buffer bindings only when buffers change
    {
        VkDescriptorSetLayout wDsLayout = waterInd.getComputeDescriptorSetLayout();
        if (wDsLayout != VK_NULL_HANDLE && cube360WaterComputeDs == VK_NULL_HANDLE) {
            VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.poolSizeCount = 1; pci.pPoolSizes = &ps; pci.maxSets = 1;
            pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(dev, &pci, nullptr, &pool) == VK_SUCCESS) {
                resources.addDescriptorPool(pool, "cubemap water compute pool");
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &wDsLayout;
                if (vkAllocateDescriptorSets(dev, &ai, &cube360WaterComputeDs) == VK_SUCCESS)
                    resources.addDescriptorSet(cube360WaterComputeDs, "cubemap water compute DS");
            }
        }
        if (cube360WaterComputeDs != VK_NULL_HANDLE) {
             std::array<VkBuffer, 10> bufs = {
                waterInd.getIndirectBuffer().buffer,
                cube360WaterCompact.buffer,
                waterInd.getBoundsBuffer().buffer,
                cube360WaterVisible.buffer,
                waterInd.getVisibleLodsScratchBuffer(),
                waterInd.getVegDummyBuffer(), // 5
                waterInd.getVegDummyBuffer(), // 6
                waterInd.getVegDummyBuffer(), // 7
                waterInd.getVegDummyBuffer(), // 8
                waterInd.getVegDummyBuffer(), // 9
            };
            if (bufs != cube360WaterComputeBuffers) {
                cube360WaterComputeBuffers = bufs;
                DescriptorWriter(dev)
                    .writeBuffer(cube360WaterComputeDs, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[0], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[1], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[2], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[3], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[4], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[5], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[6], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[7], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[8], 0, VK_WHOLE_SIZE)
                    .writeBuffer(cube360WaterComputeDs, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 bufs[9], 0, VK_WHOLE_SIZE)
                    .flush();
            }
        }
    }
}

void MyApp::postSubmit() {
    if (textureMixer) {
        textureMixer->flushPendingRequests(this);
        textureMixer->pollPendingGenerations(this);
    }

    if (brushRebuildPending) {
        brushRebuildPending = false;
        rebuildBrushScene();
    }

    if (brushApplyToScenePending) {
        brushApplyToScenePending = false;
        applyBrushToScene();
    }

    if (generateMapPending) {
        generateMapPending = false;
        generateMap();
    }

    if (loadScenePending) {
        loadScenePending = false;
        loadSceneFromFile(pendingLoadPath);
    }
}
