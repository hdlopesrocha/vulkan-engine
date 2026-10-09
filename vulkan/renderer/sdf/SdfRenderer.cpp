#include "SdfRenderer.hpp"
#include "sdf/types/SdfProxyVertex.hpp"
#include "../DescriptorAllocator.hpp"
#include "../DescriptorWriter.hpp"
#include "../RendererUtils.hpp"
#include "../../pipeline/ShaderStage.hpp"
#include "../../includes/shader/locations.hpp"
#include "../../../math/Geometry.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

// Shape ids shared with SdfSmokeConfig + the shaders: 0 cloud, 1 sphere,
// 2 cube, 3 fire (one generic flame instance on the volumetric fire path).
static constexpr int kSmokeShapeFire = 3;

// EVSM moment format of the shadow cascades (must match ShadowRenderer's
// EVSM_FORMAT); used by the grass-shadow pipeline's color attachment.
static constexpr VkFormat kSdfShadowEvsmFormat = VK_FORMAT_R32G32_SFLOAT;

SdfRenderer::SdfRenderer() = default;

SdfRenderer::~SdfRenderer() { cleanup(nullptr); }

void SdfRenderer::init(VulkanApp* app) {
    app_ = app;
    createCubeBuffers(app);
    createDescriptorSet(app);
    depthSampler = app->createSamplerLinearClamp("SdfRenderer: sceneDepthSampler");
    // Params UBOs: one per slot, created once (fixed size, never regrown).
    // SdfParamsUBO natural fields; defaults match the fragment shader
    // fallbacks so a zero time still raymarches sanely.
    params_.time = 0.0f;
    params_.maxSteps = 64.0f;
    params_.safety = 0.7f;
    params_.minStep = 0.05f;
    params_.maxStep = std::clamp(config_.lava.scale, 1.0f, 32.0f);
    params_.epsilon = 0.01f;
    params_.earlyTerm = 0.99f;
    repackDebugMode();
    renderMode_ = RenderMode::Volume; // fire-first default; generic modes via setRenderMode
    repackDebugMode();
    for (uint32_t s = 0; s < SDF_FRAMES; ++s) {
        SdfFrameSlot& slot = slots[s];
        slot.params = app->createBuffer(sizeof(SdfParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        writeSlotBinding(s, 6, slot.params, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        // Smoke state: fixed size, created once, streamed on demand.
        slot.smoke = app->createBuffer(sizeof(SmokeFragBullet), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        writeSlotBinding(s, 8, slot.smoke, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    sceneDirtySlots_.fill(true);
    paramsDirtySlots_.fill(true);
    smokeDirtySlots_.fill(true);
    // Default smoke tuning mirrors SdfEffectConfig defaults (single source:
    // SdfSmokeConfig for the cloud, SdfBulletConfig for bullet FX/gold); the
    // state block uses the canonical natural fields, packed once here.
    Smoke& st = smokeState_.tuning;
    st.growthDuration = config_.smoke.growthDuration;
    st.loopDuration = config_.smoke.loopDuration;
    st.dissipation = config_.smoke.dissipation;
    st.noiseScale = config_.smoke.noiseScale;
    st.noiseStrength = config_.smoke.noiseStrength;
    st.noiseWarp = config_.smoke.noiseWarp;
    {
        const float rad = glm::radians(config_.smoke.windAngleDeg);
        st.wind = glm::vec2(config_.smoke.windSpeed * std::cos(rad),
                            config_.smoke.windSpeed * std::sin(rad));
    }
    st.densityScale = config_.smoke.densityScale;
    st.tunnelStrength = config_.bullet.tunnelStrength;
    st.tunnelFalloff = config_.bullet.tunnelFalloff;
    st.wakeStrength = config_.bullet.wakeStrength;
    st.wakeDissipation = config_.bullet.wakeDissipation;
    st.wakeRadius = config_.bullet.wakeRadius;
    st.wakeExpansion = config_.bullet.wakeExpansion;
    st.wakeLength = config_.bullet.wakeLength;
    st.pressureRadius = config_.bullet.pressureRadius;
    st.pressureStrength = config_.bullet.pressureStrength;
    st.rippleAmp = config_.bullet.rippleAmp;
    st.rippleFreq = config_.bullet.rippleFreq;
    st.turbScale = config_.bullet.turbScale;
    st.turbStrength = config_.bullet.turbStrength;
    st.turbSpeed = config_.bullet.turbSpeed;
    st.shadowSamples = static_cast<uint32_t>(config_.smoke.shadowSamples);
    st.shadowStrength = config_.smoke.shadowStrength;
    st.heatStrength = config_.smoke.heatStrength;
    // Placement/rotation live in the generic SdfModel (instance transform),
    // never in the tuning block.
    st.shape = static_cast<uint32_t>(config_.smoke.shape);
    st.smokeColor = config_.smoke.smokeColor;
    st.goldDeep = config_.bullet.goldDeep;
    st.goldSpecPower = config_.bullet.goldSpecPower;
    st.goldBright = config_.bullet.goldBright;
    st.goldPatternScale = config_.bullet.goldPatternScale;
    st.goldSpecStrength = config_.bullet.goldSpecStrength;
    st.goldFresnelBoost = config_.bullet.goldFresnelBoost;
    st.goldWarmFloor = config_.bullet.goldWarmFloor;
    st.goldNormalDistort = config_.bullet.goldNormalDistort;
    refreshAutoBulletLocked();
    ensureSmokeScene();
    refreshMergedLocked();
    createPipeline(app);
    createShadowPipeline(app);
}

void SdfRenderer::createCubeBuffers(VulkanApp* app) {
    // Unit proxy cube in [0,1]^3; per-instance model matrix maps it onto the
    // container AABB. Position-only vertex input (ATTR_POS).
    const std::vector<SdfProxyVertex> vertices = {
        {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 1.0f}}, {{0.0f, 1.0f, 1.0f}}, {{0.0f, 1.0f, 0.0f}},
        {{1.0f, 0.0f, 0.0f}}, {{1.0f, 1.0f, 0.0f}}, {{1.0f, 1.0f, 1.0f}}, {{1.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 1.0f}},
        {{0.0f, 1.0f, 0.0f}}, {{0.0f, 1.0f, 1.0f}}, {{1.0f, 1.0f, 1.0f}}, {{1.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}}, {{0.0f, 1.0f, 0.0f}}, {{1.0f, 1.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}}, {{1.0f, 0.0f, 1.0f}}, {{1.0f, 1.0f, 1.0f}}, {{0.0f, 1.0f, 1.0f}}
    };

    std::vector<uint32_t> indices;
    indices.reserve(36);
    for (uint32_t face = 0; face < 6; ++face) {
        const uint32_t base = face * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }

    // Device-local via the async transfer path (same as DebugSDFRenderer).
    vertexBuffer = app->createDeviceLocalBufferAsync(vertices.data(),
        vertices.size() * sizeof(SdfProxyVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, nullptr);
    indexBuffer = app->createDeviceLocalBufferAsync(indices.data(),
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, nullptr);
    indexCount = static_cast<uint32_t>(indices.size());
}

void SdfRenderer::createDescriptorSet(VulkanApp* app) {
    DescriptorAllocator descAlloc{app->getDevice(), app};

    // set=1 bindings 0..9: 0 instances, 1 definitions, 2 materials,
    // 3 containers, 4 gridCells, 5 gridIndices, 6 params UBO, 7 sceneDepth,
    // 8 smoke state (tuning + bullets, fragment only), 9 waterDepth
    // (rasterized water surface, fragment only).
    VkDescriptorSetLayoutBinding bindings[10]{};
    auto storage = [&](uint32_t b, VkShaderStageFlags stages) {
        bindings[b].binding = b;
        bindings[b].descriptorCount = 1;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].stageFlags = stages;
    };
    constexpr VkShaderStageFlags kVertFrag = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    storage(0, kVertFrag); // instances read by vertex (transform) + fragment (container ref)
    storage(1, VK_SHADER_STAGE_FRAGMENT_BIT);
    storage(2, VK_SHADER_STAGE_FRAGMENT_BIT);
    storage(3, kVertFrag); // containers read by vertex (proxy transform fallback) + fragment
    storage(4, VK_SHADER_STAGE_FRAGMENT_BIT);
    storage(5, VK_SHADER_STAGE_FRAGMENT_BIT);
    storage(8, VK_SHADER_STAGE_FRAGMENT_BIT); // smoke tuning + bullets
    bindings[6].binding = 6;
    bindings[6].descriptorCount = 1;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[6].stageFlags = kVertFrag;
    bindings[7].binding = 7;
    bindings[7].descriptorCount = 1;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Water-surface depth: same kind of clamp as binding 7, gated by
    // SdfParamsUBO::waterDepthEnabled.
    bindings[9].binding = 9;
    bindings[9].descriptorCount = 1;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    descriptorSetLayout = descAlloc.createLayout(
        bindings, 10, 0, nullptr, "SdfRenderer: descriptorSetLayout");

    VkDescriptorPoolSize poolSizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * SDF_FRAMES},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * SDF_FRAMES},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * SDF_FRAMES},
    };
    descriptorPool = descAlloc.createPool(
        poolSizes, 3, SDF_FRAMES, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        "SdfRenderer: descriptorPool");

    descAlloc.allocateSets(descriptorPool, descriptorSetLayout, SDF_FRAMES, sdfSets.data(),
        "SdfRenderer: sdfSet");
}

void SdfRenderer::createPipeline(VulkanApp* app) {
    // Shaders implement the set=1 bindings 0..7 contract above. They may not
    // exist yet (parallel work); leave the pipeline null so render() cleanly
    // no-ops (clear-only) instead of crashing init.
    try {
        vertModule = app->getOrCreateShaderModule("shaders/renderer/sdf/SdfRenderer.vert.spv");
        fragModule = app->getOrCreateShaderModule("shaders/renderer/sdf/SdfRenderer.frag.spv");
    } catch (const std::exception& e) {
        std::cerr << "[SdfRenderer] shaders not available yet (" << e.what()
                  << "); pipeline deferred, render() will clear-only." << std::endl;
        return;
    }

    ShaderStage vertStage(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),
        descriptorSetLayout
    };

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_NONE; // proxy cubes viewed from inside+outside
    cfg.depthTestEnable = true;
    cfg.depthWriteEnable = true; // own depth is written for PostProcess composite-by-depth
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.colorFormats = {app->getSwapchainImageFormat()};
    cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
    auto [pipelineHandle, layoutHandle] = app->createGraphicsPipeline(
        {vertStage.info, fragStage.info},
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{0, sizeof(SdfProxyVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        },
        {
            VkVertexInputAttributeDescription{ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SdfProxyVertex, position)}
        },
        setLayouts,
        nullptr,
        cfg);

    pipeline = pipelineHandle;
    pipelineLayout = layoutHandle;
}

void SdfRenderer::createShadowPipeline(VulkanApp* app) {
    // Grass-only EVSM caster (shaders/renderer/shadow/SdfGrassShadow.*). Same
    // lazy tolerance as createPipeline: if the SPIR-V is missing, leave the
    // pipeline null so the shadow draw cleanly no-ops.
    try {
        shadowVertModule = app->getOrCreateShaderModule("shaders/renderer/shadow/SdfGrassShadow.vert.spv");
        shadowFragModule = app->getOrCreateShaderModule("shaders/renderer/shadow/SdfGrassShadow.frag.spv");
    } catch (const std::exception& e) {
        std::cerr << "[SdfRenderer] grass shadow shaders not available yet (" << e.what()
                  << "); shadow pipeline deferred." << std::endl;
        return;
    }

    ShaderStage vertStage(shadowVertModule, VK_SHADER_STAGE_VERTEX_BIT);
    ShaderStage fragStage(shadowFragModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    // Set 0 = the app's scene layout. A pipeline layout cannot index set 1
    // without a set 0; the shadow pass already has its cascade set bound
    // there (allocated with this layout) and the grass shadow shader never
    // statically uses it, so this pipeline neither binds nor reads set 0.
    std::vector<VkDescriptorSetLayout> setLayouts = {
        app->getDescriptorSetLayout(),
        descriptorSetLayout
    };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(SdfGrassShadowPC);

    GraphicsPipelineConfig cfg{};
    cfg.cullMode = VK_CULL_MODE_NONE; // proxy cubes seen from inside + outside (entry/exit fragments)
    cfg.depthTestEnable = true;
    cfg.depthWriteEnable = true;      // gl_FragDepth = grass hit depth (EVSM moment source)
    cfg.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.colorFormats = {kSdfShadowEvsmFormat};
    cfg.depthFormat = VK_FORMAT_D32_SFLOAT;
    cfg.depthBiasEnable = true;       // beginShadowRendering sets the dynamic bias
    auto [pipelineHandle, layoutHandle] = app->createGraphicsPipeline(
        {vertStage.info, fragStage.info},
        std::vector<VkVertexInputBindingDescription>{
            VkVertexInputBindingDescription{0, sizeof(SdfProxyVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        },
        {
            VkVertexInputAttributeDescription{ATTR_POS, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SdfProxyVertex, position)}
        },
        setLayouts,
        &pcRange,
        cfg);

    shadowPipeline = pipelineHandle;
    shadowPipelineLayout = layoutHandle;
}

// ─── Scene input ─────────────────────────────────────────────────────────────

void SdfRenderer::repackDebugMode() {
    params_.renderMode = static_cast<uint32_t>(renderMode_);
    params_.debugFlags = debugFlags_;
}

void SdfRenderer::setScene(const sdf_gpu::SdfScene& scene) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    lavaScene_ = scene;
    refreshMergedLocked();
}

void SdfRenderer::refreshMergedLocked() {
    // pendingScene_ is always the merge: independent emitters (lava fire,
    // rock boulders, vegetation-derived grass clumps, smoke shapes, future
    // effects) share one GPU upload and one march. It stores the canonical
    // GPU-layout structs, so rebuild() only refreshes AABBs/grids and the
    // upload memcpys the vectors verbatim.
    pendingScene_ = sdf_gpu::SdfScene::merge(
        sdf_gpu::SdfScene::merge(
            sdf_gpu::SdfScene::merge(lavaScene_, rocksScene_), grassScene_),
        smokeScene_);
    pendingScene_.rebuild();
    // Container range of the GRASS scene inside the merged scene: the merge
    // expression above concatenates containers in lava -> rocks -> grass ->
    // smoke order, so the grass containers form a contiguous range. The
    // shadow pass draws exactly this range, so fire/smoke/rock containers are
    // never rasterized. Keep in sync with the merge order above.
    grassContainerBase_ = static_cast<uint32_t>(lavaScene_.containers().size()
                                              + rocksScene_.containers().size());
    grassContainerCount_ = static_cast<uint32_t>(grassScene_.containers().size());
    stats_.containerCount = static_cast<uint32_t>(pendingScene_.containers().size());
    stats_.definitionCount = static_cast<uint32_t>(pendingScene_.definitions().size());
    stats_.materialCount = static_cast<uint32_t>(pendingScene_.materials().size());
    stats_.gridCellCount = static_cast<uint32_t>(pendingScene_.cells().size());
    sceneDirtySlots_.fill(true);
    // One-time proof of what the merged scene holds (instances per type).
    // Extended with the exact smoke-side values that decide visibility, so
    // a "sky where the cloud should be" report can be bisected from the log
    // alone (CPU state sane here + sky on screen = GPU delivery failure).
    static bool loggedOnce = false;
    if (!loggedOnce) {
        loggedOnce = true;
        size_t smokeCount = smokeScene_.instances().size();
        size_t lavaCount = lavaScene_.instances().size();
        if (!smokeScene_.containers().empty()) {
            const glm::vec3 mn = smokeScene_.containers().front().boundsMin;
            const glm::vec3 mx = smokeScene_.containers().front().boundsMax;
            fprintf(stderr, "[SdfRenderer] scene: smoke instances=%zu "
                "container=[(%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f)] lava instances=%zu\n",
                smokeCount, mn.x, mn.y, mn.z,
                mx.x, mx.y, mx.z, lavaCount);
        } else {
            fprintf(stderr, "[SdfRenderer] scene: smoke DISABLED/empty, lava instances=%zu\n",
                lavaCount);
        }
        const Smoke& st = smokeState_.tuning;
        const Bullet& b0 = smokeState_.bullets[0];
        float mden = -1.0f, mabs = -1.0f;
        if (!smokeScene_.materials().empty()) {
            mden = smokeScene_.materials()[0].density;
            mabs = smokeScene_.materials()[0].absorption;
        }
        fprintf(stderr, "[SdfRenderer] smoke state: timing=(%.2f,%.2f,%.2f) "
            "noise=(%.4f,%.2f,%.2f) wind=(%.2f,%.2f,%.2f) mat(d=%.2f,a=%.2f) "
            "march(maxStep=%.2f,maxSteps=%.0f) mode=%u flags=%u\n",
            st.growthDuration, st.loopDuration, st.dissipation,
            st.noiseScale, st.noiseStrength, st.noiseWarp,
            st.wind.x, st.wind.y, st.densityScale, mden, mabs,
            params_.maxStep, params_.maxSteps,
            static_cast<uint32_t>(renderMode_), debugFlags_);
        fprintf(stderr, "[SdfRenderer] bullet0: a=(%.0f,%.0f,%.0f,r%.1f) "
            "vel=(%.0f,%.0f,%.0f) len=%.0f c=(rEnd%.1f,loop%.1f,int%.1f,ph%.1f)\n",
            b0.start.x, b0.start.y, b0.start.z, b0.radiusStart,
            b0.velocity.x, b0.velocity.y, b0.velocity.z, b0.pathLength,
            b0.radiusEnd, b0.loopDuration, b0.intensity, b0.phase);
    }
}

void SdfRenderer::ensureSmokeScene() {
    // Static topology: 1 def/mat/container/instance (Cloud/Sphere/Cube) or
    // one flame instance (Fire). All behavior streams through the smoke
    // SSBO/instance model, so widget tweaks below never come here — only
    // position/scale/rotation/base material reshape the scene.
    if (config_.smoke.enabled) {
        // Shape rotation is part of the generic SdfModel: the instance euler
        // is the single source of rotation for the smoke shader.
        const glm::vec3 euler(glm::radians(config_.smoke.pitchDeg),
                              glm::radians(config_.smoke.yawDeg),
                              glm::radians(config_.smoke.rollDeg));
        if (config_.smoke.shape == kSmokeShapeFire) {
            // Fire shape: the shared generic flame definition, one instance,
            // rendered by the generic volumetric fire path (the same GPU path
            // as the lava-anchored flames).
            sdf_gpu::SdfScene::FlameAnchor a;
            a.pos = config_.smoke.pos;
            a.euler = euler;
            a.scale = std::max(config_.smoke.scale, 0.05f);
            a.heightScale = 1.0f;
            a.seed = config_.smoke.seed;
            a.intensity = 1.0f;
            sdf_gpu::SdfScene::FlameShape fs;
            fs.baseRadius = config_.lava.baseRadius;
            fs.height = config_.lava.height;
            fs.tipRadius = config_.lava.tipRadius;
            fs.spikiness = config_.lava.spikiness;
            fs.spikeFreq = config_.lava.spikeFreq;
            fs.density = config_.lava.flameDensity;
            smokeScene_ = sdf_gpu::SdfScene::createFireFromAnchors({a}, fs);
            // The generic anchor container pads ~3x scale above the base, but
            // the flame itself reaches (height + baseRadius) x scale plus
            // spikes. Grow the container to the instance's (rotated) bounds so
            // the single fire shape is never clipped at the container walls.
            if (!smokeScene_.containers().empty() && !smokeScene_.instances().empty()) {
                const BoundingBox ib = smokeScene_.computeInstanceBounds(smokeScene_.instances()[0]);
                auto& c = smokeScene_.containers()[0];
                c.boundsMin = glm::min(c.boundsMin, ib.getMin());
                c.boundsMax = glm::max(c.boundsMax, ib.getMax());
            }
        } else {
            smokeScene_ = sdf_gpu::SdfScene::createSmokeBomb(config_.smoke.pos, config_.smoke.scale, config_.smoke.seed);
            if (!smokeScene_.instances().empty()) {
                smokeScene_.instances()[0].rotation = euler;
            }
            // Cover the full tracer flight path for any launch angle: the
            // round starts one smoke radius out and flies bullet.length meters,
            // plus its own radius. Sized for the maxima (scale 1024 + path
            // 2560 + radius 100) so slider edits never rebuild the scene; cells
            // outside the smoke AABB stay empty and march at DDA-skip cost
            // only. XZ only: bullets fly in the shape's local XZ plane, and a
            // Y margin would overlap the terrain/lava containers below.
            if (!smokeScene_.containers().empty()) {
                auto& c = smokeScene_.containers()[0];
                const float ext = 3700.0f;
                c.boundsMin -= glm::vec3(ext, 0.0f, ext);
                c.boundsMax += glm::vec3(ext, 0.0f, ext);
            }
        }
    } else {
        smokeScene_.clear();
    }
    // Smoke material tweaks only apply to the smoke-bomb scene; the Fire
    // shape carries the flame material (density from the flame shape).
    if (config_.smoke.shape != kSmokeShapeFire && !smokeScene_.materials().empty()) {
        auto& m = smokeScene_.materials()[0];
        m.density = config_.smoke.density;
        m.absorption = config_.smoke.absorption;
        m.scattering = config_.smoke.scattering;
    }
}

bool SdfRenderer::rebuildGridIfDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    for (bool d : sceneDirtySlots_) {
        if (d) return true;
    }
    return false;
}

void SdfRenderer::updateScene() {
    rebuildGridIfDirty();
}

void SdfRenderer::updateParams(float timeSec, uint32_t frameIndex) {
    setFrame(frameIndex);
    std::lock_guard<std::mutex> lock(sceneMutex);
    params_.time = timeSec;
    lastTime_ = timeSec; // bullet birth clock (fireBullet stamps this)
    syncAutoBulletLocked(); // re-arm auto when a manual round dies (transition-only write)
    repackDebugMode();
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setRenderMode(RenderMode mode) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    renderMode_ = mode;
    repackDebugMode();
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setDebugFlags(uint32_t flags) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Preserve the smoke debug nibble (bits 4-7, owned by setSmokeDebug).
    debugFlags_ = (debugFlags_ & ~0x0Fu) | (flags & 0x0Fu);
    repackDebugMode();
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setSmokeDebug(uint32_t v) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.debugView = v & 0x0Fu;
    // Preserve the fire debug nibble (bits 0-3, owned by setDebugFlags).
    debugFlags_ = (debugFlags_ & 0x0Fu) | ((v & 0x0Fu) << 4u);
    repackDebugMode();
    paramsDirtySlots_.fill(true);
}


void SdfRenderer::setMarchParams(float maxSteps, float epsilon) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    params_.maxSteps = maxSteps;
    params_.epsilon = epsilon;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setMarchRange(float minStep, float maxStep, float earlyTermThreshold) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    params_.minStep = minStep;
    params_.maxStep = maxStep;
    params_.earlyTerm = earlyTermThreshold;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setSceneDepth(VkImageView view, VkImageLayout layout,
                                VkImage image, uint32_t width, uint32_t height) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    pendingDepthView_ = view;
    pendingDepthLayout_ = layout;
    pendingDepthImage_ = image;
    pendingDepthWidth_ = width;
    pendingDepthHeight_ = height;
}

void SdfRenderer::setWaterDepth(VkImageView view, VkImageLayout layout, bool enabled) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    pendingWaterDepthView_ = view;
    pendingWaterDepthLayout_ = layout;
    if (pendingWaterDepthEnabled_ != enabled) {
        pendingWaterDepthEnabled_ = enabled;
        params_.waterDepthEnabled = enabled ? 1.0f : 0.0f;
        paramsDirtySlots_.fill(true);
    }
}

// ─── Lava-anchored fire collection ─────────────────────────────────────────
// Mirrors VegetationRenderer::generateForChunk's brush-4 collection (area-
// weighted stochastic slots over lava triangles, deterministic per chunk)
// but emits SDF flame anchors instead of billboard instances.

void SdfRenderer::ingestLavaChunk(uintptr_t nid, const Geometry& geom) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Replace semantics: a re-published chunk (same NodeID, new version)
    // drops its old anchors first, so edits never stack duplicates.
    auto it = lavaByChunk_.find(nid);
    const bool had = (it != lavaByChunk_.end());
    lavaByChunk_.erase(nid);
    if (geom.indices.size() < 3 || geom.vertices.empty() || config_.lava.density <= 0.0f) {
        if (had) lavaDirty_ = true;
        return;
    }

    // Chunk-seeded RNG: same chunk always samples the same anchors.
    const uint32_t chunkSeed =
        static_cast<uint32_t>(nid ^ (nid >> 32)) ^ 0x9e3779b9u;
    std::mt19937 rng(chunkSeed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    // Area-weighted virtual slots over brush-4 (lava) triangles. Only the
    // LAVA fraction of a triangle earns flames (boundary triangles get
    // proportionally fewer slots), and anchors are sampled onto the lava
    // side (see below) so no flame is born on neighboring rock.
    struct LavaSlot { uint32_t i0, i1, i2; int lavaVerts; };
    std::vector<LavaSlot> triSlots;
    for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
        const uint32_t i0 = geom.indices[i + 0];
        const uint32_t i1 = geom.indices[i + 1];
        const uint32_t i2 = geom.indices[i + 2];
        if (i0 >= geom.vertices.size() || i1 >= geom.vertices.size() ||
            i2 >= geom.vertices.size())
            continue;
        const int l0 = (geom.vertices[i0].brushIndex == kLavaBrushIndex) ? 1 : 0;
        const int l1 = (geom.vertices[i1].brushIndex == kLavaBrushIndex) ? 1 : 0;
        const int l2 = (geom.vertices[i2].brushIndex == kLavaBrushIndex) ? 1 : 0;
        const int lavaVerts = l0 + l1 + l2;
        if (lavaVerts == 0) continue;
        const glm::vec3& v0 = geom.vertices[i0].position;
        const glm::vec3& v1 = geom.vertices[i1].position;
        const glm::vec3& v2 = geom.vertices[i2].position;
        const float area = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        const float lavaFrac = static_cast<float>(lavaVerts) / 3.0f;
        const float expected = std::max(0.0f, area * lavaFrac * config_.lava.density);
        uint32_t n = static_cast<uint32_t>(std::floor(expected));
        if (unit(rng) < expected - static_cast<float>(n)) ++n;
        for (uint32_t s = 0; s < n; ++s) triSlots.push_back({i0, i1, i2, lavaVerts});
    }
    if (triSlots.empty()) {
        if (had) lavaDirty_ = true;
        return; // no lava in this chunk: keep no entry (lavaChunks counts bearers)
    }

    // Shuffle so the per-chunk cap keeps a random spatial subset.
    {
        std::mt19937 srng(chunkSeed ^ 0x27d4eb2du);
        for (size_t s = triSlots.size() - 1; s > 0; --s) {
            std::uniform_int_distribution<size_t> dist(0, s);
            std::swap(triSlots[s], triSlots[dist(srng)]);
        }
    }

    // Global cap: stop adding once the scene is full (grid + SSBO stay
    // bounded; the count is visible in the SDF widget).
    size_t total = 0;
    for (const auto& kv : lavaByChunk_) total += kv.second.size();

    std::vector<sdf_gpu::SdfScene::FlameAnchor> anchors;
    anchors.reserve(std::min(triSlots.size(), kMaxLavaPerChunk));
    for (const auto& sl : triSlots) {
        if (anchors.size() >= kMaxLavaPerChunk) break;
        if (total + anchors.size() >= kMaxLavaAnchors) break;
        const Vertex& A = geom.vertices[sl.i0];
        const Vertex& B = geom.vertices[sl.i1];
        const Vertex& C = geom.vertices[sl.i2];
        // Scatter inside the triangle but keep the anchor on lava: full-lava
        // triangles sample uniformly; mixed (boundary) triangles
        // rejection-sample until the lava-vertex weight exceeds 0.5 (the
        // lava-side half/third), dropping the slot when the lava sliver is
        // too thin to hit.
        glm::vec3 p(0.0f);
        glm::vec3 bw(0.0f);
        bool placed = false;
        for (int t = 0; t < 6 && !placed; ++t) {
            float u = unit(rng), v = unit(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            const float w = 1.0f - u - v;
            if (sl.lavaVerts < 3) {
                const float lw =
                    u * ((A.brushIndex == kLavaBrushIndex) ? 1.0f : 0.0f) +
                    v * ((B.brushIndex == kLavaBrushIndex) ? 1.0f : 0.0f) +
                    w * ((C.brushIndex == kLavaBrushIndex) ? 1.0f : 0.0f);
                if (lw <= 0.5f) continue;
            }
            p = u * A.position + v * B.position + w * C.position;
            // Smooth surface normal (barycentric interpolation, like the
            // grass billboards): the flame frame tilts onto this so flames
            // stand perpendicular to the lava that spawned them.
            bw = u * A.normal + v * B.normal + w * C.normal;
            placed = true;
        }
        if (!placed) continue;
        glm::vec3 nrm(0.0f, 1.0f, 0.0f);
        {
            const float n2 = glm::dot(bw, bw);
            if (n2 > 1e-8f) nrm = bw * (1.0f / std::sqrt(n2));
        }
        const float scale = config_.lava.scale * (0.7f + 0.6f * unit(rng));
        sdf_gpu::SdfScene::FlameAnchor a;
        // Plant the flame base on the surface: the tapered flame starts at
        // the anchor (local y=0) and rises +Y, so only a small lift keeps the
        // base out of the ground through noise deformation (the buried rest
        // is occluded by the composite depth test anyway).
        a.pos = p + nrm * (0.3f * scale);
        a.euler = sdf_gpu::SdfScene::eulerAlignYToNormal(nrm);
        a.scale = scale;
        a.heightScale = 0.8f + 0.6f * unit(rng); // per-instance stretch
        a.seed = unit(rng) * 100.0f;
        a.intensity = 1.0f;
        anchors.push_back(a);
    }
    if (anchors.empty()) {
        if (had) lavaDirty_ = true;
        return;
    }
    lavaByChunk_[nid] = std::move(anchors);
    lavaDirty_ = true;
}

void SdfRenderer::removeLavaChunk(uintptr_t nid) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (lavaByChunk_.erase(nid) > 0) lavaDirty_ = true;
}

void SdfRenderer::clearLava() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!lavaByChunk_.empty()) {
        lavaByChunk_.clear();
        lavaDirty_ = true;
    }
}

bool SdfRenderer::rebuildLavaIfDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!lavaDirty_) return false;
    lavaDirty_ = false;
    std::vector<sdf_gpu::SdfScene::FlameAnchor> all;
    size_t total = 0;
    for (const auto& kv : lavaByChunk_) total += kv.second.size();
    all.reserve(total);
    // Same type on both sides now (collector stores FlameAnchors directly):
    // concatenation only, no field-by-field conversion.
    for (const auto& kv : lavaByChunk_) {
        all.insert(all.end(), kv.second.begin(), kv.second.end());
    }
    pendingScene_ = [&] {
        sdf_gpu::SdfScene::FlameShape shape;
        shape.baseRadius = config_.lava.baseRadius;
        shape.height = config_.lava.height;
        shape.tipRadius = config_.lava.tipRadius;
        shape.spikiness = config_.lava.spikiness;
        shape.spikeFreq = config_.lava.spikeFreq;
        shape.density = config_.lava.flameDensity;
        return sdf_gpu::SdfScene::createFireFromAnchors(all, shape);
    }();
    lavaScene_ = pendingScene_;
    // The Fire shape reuses the shared flame definition: keep its scene in
    // sync whenever the lava flame shape changes while Fire is selected.
    if (config_.smoke.enabled && config_.smoke.shape == kSmokeShapeFire) {
        ensureSmokeScene();
    }
    refreshMergedLocked();
    stats_.lavaAnchors = static_cast<uint32_t>(all.size());
    stats_.lavaChunks = static_cast<uint32_t>(lavaByChunk_.size());
    {
        const auto& cs = lavaScene_.containers();
        if (cs.empty()) {
            fprintf(stderr, "[SdfRenderer] lava rebuild: anchors=%zu chunks=%zu (no lava containers)\n",
                all.size(), lavaByChunk_.size());
        } else {
            const glm::vec3 mn = cs.front().boundsMin;
            const glm::vec3 mx = cs.front().boundsMax;
            fprintf(stderr, "[SdfRenderer] lava rebuild: anchors=%zu chunks=%zu "
                "container=[(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)] grid=%ux%ux%u\n",
                all.size(), lavaByChunk_.size(),
                mn.x, mn.y, mn.z, mx.x, mx.y, mx.z,
                cs.front().resX, cs.front().resY, cs.front().resZ);
        }
    }
    return true;
}

void SdfRenderer::setLavaDensity(float d) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.lava.density = std::max(0.0f, d);
}


void SdfRenderer::setLavaScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.lava.scale = std::clamp(s, 0.1f, 1024.0f);
    // Keep raymarching efficient for big flames: the adaptive step is
    // dt = clamp(d*safety, minStep, maxStep), so a 1 m maxStep would burn
    // all 64 steps just approaching a 100 m flame. Scale the ceiling with
    // the flames (sphere tracing stays conservative); interior steps stay
    // small because d is small there.
    params_.maxStep = std::clamp(config_.lava.scale, 1.0f, 32.0f);
    paramsDirtySlots_.fill(true);
}


void SdfRenderer::setLavaSpikiness(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(s, 0.0f, 1.5f);
    if (v == config_.lava.spikiness) return;
    config_.lava.spikiness = v;
    lavaDirty_ = true; // def params changed -> scene rebuild on next frame
}


void SdfRenderer::setLavaTipRadius(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 0.0f, 32.0f);
    if (v == config_.lava.tipRadius) return;
    config_.lava.tipRadius = v;
    lavaDirty_ = true;
}


void SdfRenderer::setLavaBaseRadius(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 0.05f, 32.0f);
    if (v == config_.lava.baseRadius) return;
    config_.lava.baseRadius = v;
    lavaDirty_ = true;
}


void SdfRenderer::setLavaHeight(float h) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(h, 0.5f, 10.0f);
    if (v == config_.lava.height) return;
    config_.lava.height = v;
    lavaDirty_ = true;
}


void SdfRenderer::setLavaSpikeFreq(float f) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(f, 0.5f, 6.0f);
    if (v == config_.lava.spikeFreq) return;
    config_.lava.spikeFreq = v;
    lavaDirty_ = true;
}


void SdfRenderer::setLavaFlameDensity(float d) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(d, 0.05f, 1.5f);
    if (v == config_.lava.flameDensity) return;
    config_.lava.flameDensity = v;
    lavaDirty_ = true;
}

// ─── Rock boulders (brush-7 chunks) ───────────────────────────────────────
// Mirrors the lava collector: area-weighted stochastic slots over brush-7
// triangles, deterministic per chunk, stored as per-chunk CANDIDATES at
// config().rocks.minSpacing. rebuildRocksIfDirty() decimates the candidate
// set to the live spacing and applies the live shape/material, so widget
// edits (including density) rebuild at most once per frame with no chunk
// geometry retained.

void SdfRenderer::ingestRockChunk(uintptr_t nid, const Geometry& geom) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    auto it = rocksByChunk_.find(nid);
    const bool had = (it != rocksByChunk_.end());
    rocksByChunk_.erase(nid);
    const float minSpacing = std::max(config_.rocks.minSpacing, 1.0f);
    const float perM2 = 1.0f / (minSpacing * minSpacing);
    if (geom.indices.size() < 3 || geom.vertices.empty() ||
        config_.rocks.maxAnchors == 0u) {
        if (had) rocksDirty_ = true;
        return;
    }

    // Chunk-seeded RNG: the same chunk always samples the same candidates
    // (edits re-publish the chunk, so their rocks are recreated with it).
    const uint32_t chunkSeed =
        static_cast<uint32_t>(nid ^ (nid >> 32)) ^ 0x7f4a7c15u;
    std::mt19937 rng(chunkSeed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    // Area-weighted virtual slots over brush-7 (rock) triangles. Only the
    // rock fraction of a triangle earns candidates (boundary triangles get
    // proportionally fewer slots), and candidates are sampled onto the rock
    // side so no boulder is born on neighboring grass/sand.
    struct RockSlot { uint32_t i0, i1, i2; int rockVerts; };
    std::vector<RockSlot> triSlots;
    for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
        const uint32_t i0 = geom.indices[i + 0];
        const uint32_t i1 = geom.indices[i + 1];
        const uint32_t i2 = geom.indices[i + 2];
        if (i0 >= geom.vertices.size() || i1 >= geom.vertices.size() ||
            i2 >= geom.vertices.size())
            continue;
        const int r0 = (geom.vertices[i0].brushIndex == kRockBrushIndex) ? 1 : 0;
        const int r1 = (geom.vertices[i1].brushIndex == kRockBrushIndex) ? 1 : 0;
        const int r2 = (geom.vertices[i2].brushIndex == kRockBrushIndex) ? 1 : 0;
        const int rockVerts = r0 + r1 + r2;
        if (rockVerts == 0) continue;
        const glm::vec3& v0 = geom.vertices[i0].position;
        const glm::vec3& v1 = geom.vertices[i1].position;
        const glm::vec3& v2 = geom.vertices[i2].position;
        const float area = 0.5f * glm::length(glm::cross(v1 - v0, v2 - v0));
        const float rockFrac = static_cast<float>(rockVerts) / 3.0f;
        const float expected = std::max(0.0f, area * rockFrac * perM2);
        uint32_t n = static_cast<uint32_t>(std::floor(expected));
        if (unit(rng) < expected - static_cast<float>(n)) ++n;
        for (uint32_t s = 0; s < n; ++s) triSlots.push_back({i0, i1, i2, rockVerts});
    }
    if (triSlots.empty()) {
        if (had) rocksDirty_ = true;
        return; // no rock in this chunk: keep no entry
    }

    // Shuffle so the per-chunk cap keeps a random spatial subset.
    {
        std::mt19937 srng(chunkSeed ^ 0x27d4eb2du);
        for (size_t s = triSlots.size() - 1; s > 0; --s) {
            std::uniform_int_distribution<size_t> dist(0, s);
            std::swap(triSlots[s], triSlots[dist(srng)]);
        }
    }

    // Global cap: stop adding once the retained set is full (grid + SSBO
    // stay bounded; the count is visible in the rocks widget).
    size_t total = 0;
    for (const auto& kv : rocksByChunk_) total += kv.second.size();

    std::vector<RockCandidate> candidates;
    const size_t perChunkCap = static_cast<size_t>(std::max(config_.rocks.maxPerChunk, 0));
    candidates.reserve(std::min(triSlots.size(), perChunkCap));
    for (const auto& sl : triSlots) {
        if (candidates.size() >= perChunkCap) break;
        if (total + candidates.size() >= config_.rocks.maxAnchors) break;
        const Vertex& A = geom.vertices[sl.i0];
        const Vertex& B = geom.vertices[sl.i1];
        const Vertex& C = geom.vertices[sl.i2];
        // Scatter inside the triangle but keep the candidate on rock:
        // full-rock triangles sample uniformly; mixed (boundary) triangles
        // rejection-sample until the rock-vertex weight exceeds 0.5 (the
        // rock-side half/third), dropping the slot when the rock sliver is
        // too thin to hit.
        glm::vec3 p(0.0f);
        glm::vec3 bw(0.0f);
        bool placed = false;
        for (int t = 0; t < 6 && !placed; ++t) {
            float u = unit(rng), v = unit(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            const float w = 1.0f - u - v;
            if (sl.rockVerts < 3) {
                const float lw =
                    u * ((A.brushIndex == kRockBrushIndex) ? 1.0f : 0.0f) +
                    v * ((B.brushIndex == kRockBrushIndex) ? 1.0f : 0.0f) +
                    w * ((C.brushIndex == kRockBrushIndex) ? 1.0f : 0.0f);
                if (lw <= 0.5f) continue;
            }
            p = u * A.position + v * B.position + w * C.position;
            // Smooth surface normal (barycentric interpolation): the boulder
            // sits along this normal so it stands perpendicular to the rock
            // face (overhangs included).
            bw = u * A.normal + v * B.normal + w * C.normal;
            placed = true;
        }
        if (!placed) continue;
        RockCandidate c;
        c.pos = p;
        {
            const float n2 = glm::dot(bw, bw);
            if (n2 > 1e-8f) c.normal = bw * (1.0f / std::sqrt(n2));
        }
        // Per-candidate variation keys: sampled once here, consumed at every
        // rebuild so size/rotation/density edits keep each rock's identity.
        c.euler = glm::vec3(unit(rng) * 6.2831853f,
                            unit(rng) * 6.2831853f,
                            unit(rng) * 6.2831853f);
        c.scaleVar = unit(rng);
        c.seed = unit(rng) * 100.0f;
        c.decim = unit(rng);
        candidates.push_back(c);
    }
    if (candidates.empty()) {
        if (had) rocksDirty_ = true;
        return;
    }
    rocksByChunk_[nid] = std::move(candidates);
    rocksDirty_ = true;
}

void SdfRenderer::removeRockChunk(uintptr_t nid) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (rocksByChunk_.erase(nid) > 0) rocksDirty_ = true;
}

void SdfRenderer::clearRocks() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!rocksByChunk_.empty()) {
        rocksByChunk_.clear();
        rocksDirty_ = true;
    }
}

void SdfRenderer::markRocksDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    rocksDirty_ = true;
}

bool SdfRenderer::rebuildRocksIfDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!rocksDirty_) return false;
    rocksDirty_ = false;

    // Density decimation: candidates were ingested at minSpacing; keep the
    // fraction (minSpacing / spacing)^2 so the expected count is one rock
    // per spacing x spacing m of rock surface. The per-candidate key makes
    // the kept set stable while the spacing slider moves.
    const float minSpacing = std::max(config_.rocks.minSpacing, 1.0f);
    const float spacing = std::max(config_.rocks.spacing, minSpacing);
    float keepRatio = (minSpacing * minSpacing) / (spacing * spacing);
    if (!config_.rocks.enabled) keepRatio = 0.0f;
    keepRatio = std::clamp(keepRatio, 0.0f, 1.0f);

    const float var = std::clamp(config_.rocks.scaleVariation, 0.0f, 1.0f);
    const float embed = std::clamp(config_.rocks.embed, 0.0f, 0.95f);
    const float baseScale = std::max(config_.rocks.scale, 0.1f);

    std::vector<sdf_gpu::SdfScene::RockAnchor> all;
    size_t candidateCount = 0;
    for (const auto& kv : rocksByChunk_) candidateCount += kv.second.size();
    all.reserve(static_cast<size_t>(candidateCount * keepRatio) + rocksByChunk_.size());
    for (const auto& kv : rocksByChunk_) {
        for (const RockCandidate& c : kv.second) {
            if (c.decim >= keepRatio) continue;
            const float scale = baseScale * (1.0f - var + 2.0f * var * c.scaleVar);
            sdf_gpu::SdfScene::RockAnchor a;
            // Sink the sphere by `embed` of its radius so a boulder sits in
            // the ground instead of floating tangent to it.
            a.pos = c.pos + c.normal * (scale * (1.0f - embed));
            a.euler = c.euler;
            a.scale = scale;
            a.seed = c.seed;
            a.intensity = 1.0f;
            all.push_back(a);
        }
    }

    rocksScene_ = [&] {
        sdf_gpu::SdfScene::RockShape shape;
        shape.noiseScale = config_.rocks.noiseScale;
        shape.noiseAmplitude = config_.rocks.noiseAmplitude;
        shape.textureLayer = config_.rocks.textureLayer;
        shape.textureTiling = config_.rocks.textureTiling;
        shape.roughness = config_.rocks.roughness;
        shape.metallic = config_.rocks.metallic;
        shape.tint = config_.rocks.tint;
        return sdf_gpu::SdfScene::createRocksFromAnchors(all, shape);
    }();
    refreshMergedLocked();
    stats_.rockAnchors = static_cast<uint32_t>(all.size());
    stats_.rockChunks = static_cast<uint32_t>(rocksByChunk_.size());
    if (!all.empty()) {
        const auto& cs = rocksScene_.containers();
        if (!cs.empty()) {
            const glm::vec3 mn = cs.front().boundsMin;
            const glm::vec3 mx = cs.front().boundsMax;
            fprintf(stderr, "[SdfRenderer] rock rebuild: rocks=%zu candidates=%zu "
                "chunks=%zu spacing=%.0f container=[(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)] grid=%ux%ux%u\n",
                all.size(), candidateCount, rocksByChunk_.size(), spacing,
                mn.x, mn.y, mn.z, mx.x, mx.y, mx.z,
                cs.front().resX, cs.front().resY, cs.front().resZ);
        }
    }
    return true;
}

void SdfRenderer::setRocksEnabled(bool on) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (on == config_.rocks.enabled) return;
    config_.rocks.enabled = on;
    rocksDirty_ = true;
}

void SdfRenderer::setRockSpacing(float m) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(m, config_.rocks.minSpacing, 4096.0f);
    if (v == config_.rocks.spacing) return;
    config_.rocks.spacing = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockMaxPerChunk(int n) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.rocks.maxPerChunk = std::clamp(n, 0, 256);
}

void SdfRenderer::setRockScale(float m) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(m, 0.5f, 512.0f);
    if (v == config_.rocks.scale) return;
    config_.rocks.scale = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockScaleVariation(float f) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(f, 0.0f, 1.0f);
    if (v == config_.rocks.scaleVariation) return;
    config_.rocks.scaleVariation = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockEmbed(float f) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(f, 0.0f, 0.95f);
    if (v == config_.rocks.embed) return;
    config_.rocks.embed = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockNoiseScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(s, 0.05f, 32.0f);
    if (v == config_.rocks.noiseScale) return;
    config_.rocks.noiseScale = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockNoiseAmplitude(float a) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(a, 0.0f, 1.0f);
    if (v == config_.rocks.noiseAmplitude) return;
    config_.rocks.noiseAmplitude = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockTextureLayer(float layer) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(layer, -1.0f, 255.0f);
    if (v == config_.rocks.textureLayer) return;
    config_.rocks.textureLayer = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockTextureTiling(float m) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(m, 1.0f, 1024.0f);
    if (v == config_.rocks.textureTiling) return;
    config_.rocks.textureTiling = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockRoughness(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 0.02f, 1.0f);
    if (v == config_.rocks.roughness) return;
    config_.rocks.roughness = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockMetallic(float m) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(m, 0.0f, 1.0f);
    if (v == config_.rocks.metallic) return;
    config_.rocks.metallic = v;
    rocksDirty_ = true;
}

void SdfRenderer::setRockTint(const glm::vec3& rgb) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const glm::vec3 v = glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(4.0f));
    if (v == config_.rocks.tint) return;
    config_.rocks.tint = v;
    rocksDirty_ = true;
}

// ─── Grass clumps (existing vegetation instances) ─────────────────────────
// Grass anchors stream in 1:1 from VegetationRenderer as vegetation chunks
// publish: the existing vegetation instance (position + billboard type/biome
// + smooth surface normal) is the source of truth, so this collector never
// places grass itself. The caps only bound the SDF-side set (SSBO + grid);
// the shader expands each retained anchor into many procedural blades.

void SdfRenderer::ingestGrassChunk(uintptr_t nid, std::vector<sdf_gpu::GrassAnchor> anchors) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Replace semantics: a re-published vegetation chunk (same NodeID, new
    // version) drops its old clumps first, so re-streaming never stacks
    // duplicates.
    auto it = grassByChunk_.find(nid);
    const bool had = (it != grassByChunk_.end());
    grassByChunk_.erase(nid);
    if (anchors.empty() || config_.grass.maxPerChunk <= 0 || config_.grass.maxAnchors == 0u) {
        if (had) grassDirty_ = true;
        return;
    }

    // Per-chunk cap: the vegetation generator already thins dense chunks
    // with shuffled area-weighted slots, so a uniform stride over the
    // streamed order keeps a deterministic, spatially even subset.
    const size_t cap = std::min(anchors.size(),
                                static_cast<size_t>(config_.grass.maxPerChunk));
    std::vector<sdf_gpu::GrassAnchor> kept;
    kept.reserve(cap);
    if (anchors.size() > cap) {
        const double stride = static_cast<double>(anchors.size()) / static_cast<double>(cap);
        for (size_t i = 0; i < cap; ++i) {
            kept.push_back(anchors[static_cast<size_t>(static_cast<double>(i) * stride)]);
        }
    } else {
        kept = std::move(anchors);
    }

    // Global cap: stop adding once the retained set is full (mirrors the
    // lava/rock collectors; the count is visible in the SDF widget).
    size_t total = 0;
    for (const auto& kv : grassByChunk_) total += kv.second.size();
    if (total >= config_.grass.maxAnchors) {
        if (had) grassDirty_ = true;
        return;
    }
    if (total + kept.size() > config_.grass.maxAnchors) {
        kept.resize(config_.grass.maxAnchors - total);
    }
    grassByChunk_[nid] = std::move(kept);
    grassDirty_ = true;
}

void SdfRenderer::removeGrassChunk(uintptr_t nid) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (grassByChunk_.erase(nid) > 0) grassDirty_ = true;
}

void SdfRenderer::clearGrass() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!grassByChunk_.empty()) {
        grassByChunk_.clear();
        grassDirty_ = true;
    }
}

void SdfRenderer::markGrassDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    grassDirty_ = true;
}

bool SdfRenderer::rebuildGrassIfDirty() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (!grassDirty_) return false;
    grassDirty_ = false;

    // Flatten the retained per-chunk anchors (the scene rebuild shares the
    // same coalesced path as lava/rocks: at most one flatten per frame).
    std::vector<sdf_gpu::GrassAnchor> all;
    size_t tracked = 0;
    for (const auto& kv : grassByChunk_) tracked += kv.second.size();
    all.reserve(tracked);
    if (config_.grass.enabled) {
        for (const auto& kv : grassByChunk_) {
            all.insert(all.end(), kv.second.begin(), kv.second.end());
        }
        if (all.size() > config_.grass.maxAnchors) all.resize(config_.grass.maxAnchors);
    }

    grassScene_ = [&] {
        sdf_gpu::GrassShape shape;
        shape.clumpRadius = config_.grass.clumpRadius;
        shape.bladeHeight = config_.grass.bladeHeight;
        shape.bladeWidth = config_.grass.bladeWidth;
        shape.bladeCount = config_.grass.bladeCount;
        shape.curvature = config_.grass.curvature;
        shape.maxLean = config_.grass.maxLean;
        shape.windGain = config_.grass.windGain;
        shape.tipWidth = config_.grass.tipWidth;
        shape.regionSize = config_.grass.regionSize;
        shape.cellSize = config_.grass.cellSize;
        shape.roughness = config_.grass.roughness;
        shape.tint = config_.grass.tint;
        return sdf_gpu::SdfScene::createGrassFromAnchors(all, shape);
    }();
    refreshMergedLocked();
    stats_.grassAnchors = static_cast<uint32_t>(all.size());
    stats_.grassChunks = static_cast<uint32_t>(grassByChunk_.size());
    if (!all.empty()) {
        const auto& cs = grassScene_.containers();
        if (!cs.empty()) {
            fprintf(stderr, "[SdfRenderer] grass rebuild: clumps=%zu tracked=%zu "
                "chunks=%zu regions=%zu grid=%ux%ux%u\n",
                all.size(), tracked, grassByChunk_.size(), cs.size(),
                cs.front().resX, cs.front().resY, cs.front().resZ);
        }
    } else if (tracked > 0) {
        fprintf(stderr, "[SdfRenderer] grass DISABLED: clumps=%zu tracked (candidates retained)\n",
            tracked);
    }
    return true;
}

void SdfRenderer::setGrassEnabled(bool on) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (on == config_.grass.enabled) return;
    config_.grass.enabled = on;
    grassDirty_ = true;
}

void SdfRenderer::setGrassMaxPerChunk(int n) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Applies to newly ingested chunks (retained sets keep their size).
    config_.grass.maxPerChunk = std::clamp(n, 0, 1024);
}

void SdfRenderer::setGrassClumpRadius(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 0.05f, 2.0f);
    if (v == config_.grass.clumpRadius) return;
    config_.grass.clumpRadius = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassBladeHeight(float h) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(h, 0.05f, 4.0f);
    if (v == config_.grass.bladeHeight) return;
    config_.grass.bladeHeight = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassBladeWidth(float w) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(w, 0.001f, 0.5f);
    if (v == config_.grass.bladeWidth) return;
    config_.grass.bladeWidth = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassBladeCount(int n) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const int v = std::clamp(n, 1, 64);
    if (v == config_.grass.bladeCount) return;
    config_.grass.bladeCount = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassCurvature(float c) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(c, 0.0f, 1.5f);
    if (v == config_.grass.curvature) return;
    config_.grass.curvature = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassMaxLean(float rad) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(rad, 0.0f, 1.4f);
    if (v == config_.grass.maxLean) return;
    config_.grass.maxLean = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassWindGain(float g) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(g, 0.0f, 2.0f);
    if (v == config_.grass.windGain) return;
    config_.grass.windGain = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassTipWidth(float f) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(f, 0.05f, 1.0f);
    if (v == config_.grass.tipWidth) return;
    config_.grass.tipWidth = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassRoughness(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 0.02f, 1.0f);
    if (v == config_.grass.roughness) return;
    config_.grass.roughness = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassTint(const glm::vec3& rgb) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const glm::vec3 v = glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(4.0f));
    if (v == config_.grass.tint) return;
    config_.grass.tint = v;
    grassDirty_ = true;
}

void SdfRenderer::setGrassShadowLodScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // No scene rebuild: the shadow pass reads this per frame (push constant).
    config_.grass.shadowLodScale = std::clamp(s, 4.0f, 120.0f);
}

void SdfRenderer::setGrassImpostorStart(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(s, 0.0f, 256.0f);
    if (v == config_.grass.impostorStart) return;
    config_.grass.impostorStart = v;
    // Keep the band ordered (start <= full).
    config_.grass.impostorFull = std::max(config_.grass.impostorFull, v);
    params_.impostorStart = config_.grass.impostorStart;
    params_.impostorFull = config_.grass.impostorFull;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setGrassImpostorFull(float f) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(f, 0.0f, 512.0f);
    if (v == config_.grass.impostorFull) return;
    config_.grass.impostorFull = std::max(v, config_.grass.impostorStart);
    params_.impostorFull = config_.grass.impostorFull;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setGrassImpostorsOnly(bool on) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (on) {
        // Degenerate band: any camScale >= 0 takes the hard impostor-only
        // path (sdGrassClump's full >= start == 0 test), so every clump is
        // rendered as its SDF impostor for inspection.
        config_.grass.impostorStart = 0.0f;
        config_.grass.impostorFull = 0.0f;
    } else {
        config_.grass.impostorStart = 48.0f;
        config_.grass.impostorFull = 80.0f;
    }
    params_.impostorStart = config_.grass.impostorStart;
    params_.impostorFull = config_.grass.impostorFull;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setRaycastPixelSize(int px) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const int v = std::clamp(px, 1, 8);
    if (v == raycastPixelSize_) return;
    raycastPixelSize_ = v;
    params_.raycastPixelSize = static_cast<float>(v);
    paramsDirtySlots_.fill(true);
}

// ─── Smoke bomb + bullets ─────────────────────────────────────────────────
// Scene-affecting setters rebuild the static smoke topology (cheap: 1 def /
// mat / container / instance) and re-merge; tuning setters only stream the
// smoke SSBO (no re-flatten).

void SdfRenderer::markSmokeSSBO() {
    smokeDirtySlots_.fill(true);
}

void SdfRenderer::refreshAutoBulletLocked() {
    const float rad = glm::radians(config_.bullet.angleDeg);
    const glm::vec3 dirW(std::cos(rad), 0.0f, std::sin(rad));
    const glm::vec3 startW = config_.smoke.pos - dirW * config_.smoke.scale;
    // The smoke effect is evaluated in the shape's local frame (the generic
    // SdfModel owns the transform), so bullets are packed there too: the
    // shader never transforms a bullet. World behavior is unchanged — the
    // stored local direction/start map back onto the world direction via the
    // instance rotation.
    const glm::mat3 rot = smokeRotLocked();
    const glm::mat3 rInv = glm::transpose(rot);
    const glm::vec3 dir = rInv * dirW;
    const glm::vec3 start = rInv * (startW - config_.smoke.pos);
    // Pack the same instance transform the bullets live in: the fragment
    // shader's analytic gold-tracer test runs in this local frame (world
    // camera and local bullet coordinates never meet).
    smokeState_.worldPos = config_.smoke.pos;
    smokeState_.worldScale = 1.0f; // createSmokeBomb bakes `scale` into params0
    smokeState_.rotCol0 = rot[0];
    smokeState_.rotCol1 = rot[1];
    smokeState_.rotCol2 = rot[2];
    // The analytic tracer only exists for the smoke-bomb shapes; Fire has no
    // bullets and disabled smoke has no scene to composite against.
    smokeState_.tracerActive =
        (config_.smoke.enabled && config_.smoke.shape != kSmokeShapeFire) ? 1.0f : 0.0f;
    // Canonical Bullet: the round starts on the smoke surface (one radius
    // out) and is PARKED there for the whole expansion; phase = growth
    // duration is the movement start within each loop. intensity 0 = empty.
    Bullet& b = smokeState_.bullets[0];
    b.start = start;
    b.radiusStart = config_.bullet.radius;
    b.velocity = dir * config_.bullet.speed;
    b.pathLength = config_.bullet.length;
    b.radiusEnd = config_.bullet.finalRadius;
    b.loopDuration = config_.bullet.loopDuration;
    b.intensity = 1.0f;
    b.phase = config_.smoke.growthDuration;
    syncAutoBulletLocked(); // applies autoFire + single-flight (marks dirty on change)
    markSmokeSSBO();
}

glm::mat3 SdfRenderer::smokeRotLocked() const {
    // Mirrors sdfEulerMat (R = Rx * Ry * Rz) in sdf_ops.glsl; the widget's
    // yaw/pitch/roll map to the Y/X/Z euler angles. Caller holds sceneMutex.
    const float yaw = glm::radians(config_.smoke.yawDeg);
    const float pitch = glm::radians(config_.smoke.pitchDeg);
    const float roll = glm::radians(config_.smoke.rollDeg);
    const float cx = std::cos(pitch);
    const float sx = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float cz = std::cos(roll);
    const float sz = std::sin(roll);
    const glm::mat3 rx(1.0f, 0.0f, 0.0f, 0.0f, cx, sx, 0.0f, -sx, cx);
    const glm::mat3 ry(cy, 0.0f, -sy, 0.0f, 1.0f, 0.0f, sy, 0.0f, cy);
    const glm::mat3 rz(cz, sz, 0.0f, -sz, cz, 0.0f, 0.0f, 0.0f, 1.0f);
    return rx * ry * rz;
}

bool SdfRenderer::anyManualBulletLiveLocked() const {
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i) {
        const Bullet& b = smokeState_.bullets[i];
        if (b.intensity <= 0.0f) continue;
        const float speed = std::max(glm::length(b.velocity), 1e-3f);
        const float travelTime = std::max(b.pathLength, 1e-3f) / speed;
        const float age = (b.loopDuration > 0.0f)
            ? std::fmod(lastTime_ - b.phase, b.loopDuration)
            : (lastTime_ - b.phase);
        // Same live window as the shader (flight + 8 s wake grace).
        if (age >= 0.0f && age <= travelTime + 8.0f) return true;
    }
    return false;
}

void SdfRenderer::syncAutoBulletLocked() {
    // Single-flight: a live manual round suppresses the auto bullet until
    // it dies, so at most one bullet (auto XOR manual) is ever in flight.
    // Only writes on transitions, so steady state costs nothing.
    const float want = (config_.bullet.autoFire && !anyManualBulletLiveLocked()) ? 1.0f : 0.0f;
    if (smokeState_.bullets[0].intensity != want) {
        smokeState_.bullets[0].intensity = want;
        markSmokeSSBO();
    }
}

void SdfRenderer::setSmokeEnabled(bool e) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (e == config_.smoke.enabled) return;
    config_.smoke.enabled = e;
    ensureSmokeScene();
    // Re-pack the tracer transform/gate: a disabled smoke has no scene to
    // composite the gold round against, so tracerActive drops to 0.
    refreshAutoBulletLocked();
    refreshMergedLocked();
}


void SdfRenderer::setSmokePosition(const glm::vec3& p) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (p == config_.smoke.pos) return;
    config_.smoke.pos = p;
    ensureSmokeScene();
    refreshAutoBulletLocked();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeScale(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 8.0f, 1024.0f);
    if (v == config_.smoke.scale) return;
    config_.smoke.scale = v;
    ensureSmokeScene();
    refreshAutoBulletLocked();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeGrowthDuration(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Growth must fit inside the loop, or expansion never completes and the
    // bullet-start invariant below cannot hold.
    config_.smoke.growthDuration = std::clamp(std::min(s, config_.smoke.loopDuration), 0.5f, 60.0f);
    smokeState_.tuning.growthDuration = config_.smoke.growthDuration;
    // The auto round starts only after expansion: its phase IS the growth
    // duration, so re-stamp it whenever growth changes (otherwise a tuned
    // growth leaves the bullet crossing an unformed cloud).
    refreshAutoBulletLocked();
    markSmokeSSBO();
}


void SdfRenderer::setSmokeLoopDuration(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.loopDuration = std::clamp(s, 2.0f, 120.0f);
    smokeState_.tuning.loopDuration = config_.smoke.loopDuration;
    // Keep growth inside the shortened loop for the same invariant.
    if (config_.smoke.growthDuration > config_.smoke.loopDuration) {
        config_.smoke.growthDuration = config_.smoke.loopDuration;
        smokeState_.tuning.growthDuration = config_.smoke.growthDuration;
        refreshAutoBulletLocked();
    }
    markSmokeSSBO();
}


void SdfRenderer::setSmokeDissipation(float d) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.dissipation = std::clamp(d, 0.0f, 2.0f);
    smokeState_.tuning.dissipation = config_.smoke.dissipation;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseScale = std::clamp(s, 0.001f, 0.2f);
    smokeState_.tuning.noiseScale = config_.smoke.noiseScale;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseStrength(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseStrength = std::clamp(s, 0.0f, 2.0f);
    smokeState_.tuning.noiseStrength = config_.smoke.noiseStrength;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseWarp(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseWarp = std::clamp(s, 0.0f, 2.0f);
    smokeState_.tuning.noiseWarp = config_.smoke.noiseWarp;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeWind(float speed, float angleDeg) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.windSpeed = std::clamp(speed, 0.0f, 60.0f);
    config_.smoke.windAngleDeg = angleDeg;
    const float rad = glm::radians(config_.smoke.windAngleDeg);
    smokeState_.tuning.wind.x = config_.smoke.windSpeed * std::cos(rad);
    smokeState_.tuning.wind.y = config_.smoke.windSpeed * std::sin(rad);
    markSmokeSSBO();
}



void SdfRenderer::setSmokeDensityScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.densityScale = std::clamp(s, 0.0f, 3.0f);
    smokeState_.tuning.densityScale = config_.smoke.densityScale;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeDensity(float d) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(d, 0.05f, 2.0f);
    if (v == config_.smoke.density) return;
    config_.smoke.density = v;
    ensureSmokeScene();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeAbsorption(float a) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(a, 0.0f, 3.0f);
    if (v == config_.smoke.absorption) return;
    config_.smoke.absorption = v;
    ensureSmokeScene();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeScattering(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(s, 0.0f, 8.0f);
    if (v == config_.smoke.scattering) return;
    config_.smoke.scattering = v;
    ensureSmokeScene();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeTunnel(float strength, float falloff) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.tunnelStrength = std::clamp(strength, 0.0f, 1.0f);
    smokeState_.tuning.tunnelStrength = config_.bullet.tunnelStrength;
    config_.bullet.tunnelFalloff = std::clamp(falloff, 0.5f, 100.0f);
    smokeState_.tuning.tunnelFalloff = config_.bullet.tunnelFalloff;
    markSmokeSSBO();
}



void SdfRenderer::setSmokeWake(float strength, float radius, float expansion, float length, float dissipation) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.wakeStrength = std::clamp(strength, 0.0f, 2.0f);
    smokeState_.tuning.wakeStrength = config_.bullet.wakeStrength;
    config_.bullet.wakeRadius = std::clamp(radius, 0.5f, 100.0f);
    smokeState_.tuning.wakeRadius = config_.bullet.wakeRadius;
    config_.bullet.wakeExpansion = std::clamp(expansion, 0.0f, 1.0f);
    smokeState_.tuning.wakeExpansion = config_.bullet.wakeExpansion;
    config_.bullet.wakeLength = std::clamp(length, 1.0f, 500.0f);
    smokeState_.tuning.wakeLength = config_.bullet.wakeLength;
    config_.bullet.wakeDissipation = std::clamp(dissipation, 0.05f, 3.0f);
    smokeState_.tuning.wakeDissipation = config_.bullet.wakeDissipation;
    markSmokeSSBO();
}






void SdfRenderer::setSmokeShock(float radius, float strength, float rippleAmp, float rippleFreq) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.pressureRadius = std::clamp(radius, 0.5f, 150.0f);
    smokeState_.tuning.pressureRadius = config_.bullet.pressureRadius;
    config_.bullet.pressureStrength = std::clamp(strength, 0.0f, 30.0f);
    smokeState_.tuning.pressureStrength = config_.bullet.pressureStrength;
    config_.bullet.rippleAmp = std::clamp(rippleAmp, 0.0f, 0.5f);
    smokeState_.tuning.rippleAmp = config_.bullet.rippleAmp;
    config_.bullet.rippleFreq = std::clamp(rippleFreq, 0.5f, 40.0f);
    smokeState_.tuning.rippleFreq = config_.bullet.rippleFreq;
    markSmokeSSBO();
}

void SdfRenderer::setSmokeHeatStrength(float strength) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.heatStrength = std::clamp(strength, 0.0f, 3.0f);
    smokeState_.tuning.heatStrength = config_.smoke.heatStrength;
    markSmokeSSBO();
}






void SdfRenderer::setSmokeTurbulence(float scale, float strength, float speed) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.turbScale = std::clamp(scale, 0.001f, 1.0f);
    smokeState_.tuning.turbScale = config_.bullet.turbScale;
    config_.bullet.turbStrength = std::clamp(strength, 0.0f, 10.0f);
    smokeState_.tuning.turbStrength = config_.bullet.turbStrength;
    config_.bullet.turbSpeed = std::clamp(speed, 0.0f, 10.0f);
    smokeState_.tuning.turbSpeed = config_.bullet.turbSpeed;
    markSmokeSSBO();
}


void SdfRenderer::setBulletGold(const glm::vec3& deep, const glm::vec3& bright,
                                float specPower, float specStrength, float fresnelBoost,
                                float warmFloor, float patternScale, float normalDistort) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.goldDeep = glm::clamp(deep, glm::vec3(0.0f), glm::vec3(4.0f));
    config_.bullet.goldBright = glm::clamp(bright, glm::vec3(0.0f), glm::vec3(4.0f));
    config_.bullet.goldSpecPower = std::clamp(specPower, 1.0f, 512.0f);
    config_.bullet.goldSpecStrength = std::clamp(specStrength, 0.0f, 10.0f);
    config_.bullet.goldFresnelBoost = std::clamp(fresnelBoost, 0.0f, 4.0f);
    config_.bullet.goldWarmFloor = std::clamp(warmFloor, 0.0f, 1.0f);
    config_.bullet.goldPatternScale = std::clamp(patternScale, 0.05f, 8.0f);
    config_.bullet.goldNormalDistort = std::clamp(normalDistort, 0.0f, 2.0f);
    smokeState_.tuning.goldDeep = config_.bullet.goldDeep;
    smokeState_.tuning.goldSpecPower = config_.bullet.goldSpecPower;
    smokeState_.tuning.goldBright = config_.bullet.goldBright;
    smokeState_.tuning.goldPatternScale = config_.bullet.goldPatternScale;
    smokeState_.tuning.goldSpecStrength = config_.bullet.goldSpecStrength;
    smokeState_.tuning.goldFresnelBoost = config_.bullet.goldFresnelBoost;
    smokeState_.tuning.goldWarmFloor = config_.bullet.goldWarmFloor;
    smokeState_.tuning.goldNormalDistort = config_.bullet.goldNormalDistort;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeColor(const glm::vec3& rgb) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.smokeColor = glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(4.0f));
    smokeState_.tuning.smokeColor = config_.smoke.smokeColor;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeShape(int shape) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const int v = std::clamp(shape, 0, kSmokeShapeFire);
    if (v == config_.smoke.shape) return;
    const bool wasFire = (config_.smoke.shape == kSmokeShapeFire);
    const bool isFire = (v == kSmokeShapeFire);
    config_.smoke.shape = v;
    smokeState_.tuning.shape = static_cast<uint32_t>(v);
    // Cloud/Sphere/Cube only change the shader-side envelope/density; the
    // Fire shape swaps the scene primitive (flame instead of smoke bomb), so
    // rebuild the smoke scene on either side of that transition.
    if (isFire || wasFire) {
        ensureSmokeScene();
        refreshMergedLocked();
    }
    // The tracer gate follows the shape (Fire has no bullets).
    refreshAutoBulletLocked();
    markSmokeSSBO();
}


void SdfRenderer::setSmokeRotation(float yawDeg, float pitchDeg, float rollDeg) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.yawDeg = yawDeg;
    config_.smoke.pitchDeg = pitchDeg;
    config_.smoke.rollDeg = rollDeg;
    // Rotation lives in the generic SdfModel (instance euler): rebuild the
    // instance and re-pack the local-space bullets so the world behavior is
    // unchanged. No shapeParams writes here anymore.
    ensureSmokeScene();
    refreshAutoBulletLocked();
    refreshMergedLocked();
}




void SdfRenderer::setSmokeShadow(int samples, float strength) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.shadowSamples = std::clamp(samples, 1, 8);
    smokeState_.tuning.shadowSamples = static_cast<uint32_t>(config_.smoke.shadowSamples);
    config_.smoke.shadowStrength = std::clamp(strength, 0.0f, 1.0f);
    smokeState_.tuning.shadowStrength = config_.smoke.shadowStrength;
    markSmokeSSBO();
}



void SdfRenderer::setBulletDefaults(float radiusStart, float radiusEnd, float speed,
                                    float length, float angleDeg, float loopDuration) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.radius = std::clamp(radiusStart, 0.5f, 100.0f);
    config_.bullet.finalRadius = std::clamp(radiusEnd, 0.5f, 100.0f);
    config_.bullet.speed = std::clamp(speed, 1.0f, 1000.0f);
    config_.bullet.length = std::clamp(length, 10.0f, 2000.0f);
    config_.bullet.angleDeg = angleDeg;
    config_.bullet.loopDuration = std::clamp(loopDuration, 1.0f, 60.0f);
    refreshAutoBulletLocked();
}





void SdfRenderer::setAutoFire(bool on) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.autoFire = on;
    refreshAutoBulletLocked();
}


void SdfRenderer::fireBullet(float angleDeg) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float rad = glm::radians(angleDeg);
    const glm::vec3 dirW(std::cos(rad), 0.0f, std::sin(rad));
    const glm::vec3 startW = config_.smoke.pos - dirW * config_.smoke.scale;
    // Packed in the shape's local frame, like the auto round above.
    const glm::mat3 rInv = glm::transpose(smokeRotLocked());
    const glm::vec3 dir = rInv * dirW;
    const glm::vec3 start = rInv * (startW - config_.smoke.pos);
    // Single-flight: exactly one bullet at a time. Clear any previous manual
    // round (rapid firing can't stack tunnels) and stamp the single manual
    // slot; slot 0 stays the auto-loop template, suppressed below while the
    // manual round lives.
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i)
        smokeState_.bullets[i].intensity = 0.0f;
    Bullet& m = smokeState_.bullets[1];
    m.start = start;
    // One-shot: loopDuration 0 disables the per-bullet loop in the shader, so
    // the round flies once (born now: phase = lastTime_) and disappears;
    // intensity 1 marks it in flight.
    m.radiusStart = config_.bullet.radius;
    m.velocity = dir * config_.bullet.speed;
    m.pathLength = config_.bullet.length;
    m.radiusEnd = config_.bullet.finalRadius;
    m.loopDuration = 0.0f;
    m.intensity = 1.0f;
    m.phase = lastTime_;
    syncAutoBulletLocked();
    markSmokeSSBO();
}

void SdfRenderer::clearBullets() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i) {
        smokeState_.bullets[i].intensity = 0.0f; // intensity 0 = empty
    }
    markSmokeSSBO();
}

uint32_t SdfRenderer::bulletSlotsUsed() const {
    std::lock_guard<std::mutex> lock(sceneMutex);
    // Count only bullets still in flight (or within the wake grace window):
    // one-shot manual bullets stay in their slot after the pass, so a raw
    // intensity test would over-report forever.
    uint32_t n = 0;
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i) {
        const Bullet& b = smokeState_.bullets[i];
        if (b.intensity <= 0.0f) continue;
        const float speed = std::max(glm::length(b.velocity), 1e-3f);
        const float pathLen = std::max(b.pathLength, 1e-3f);
        const float travelTime = pathLen / speed;
        const float age = (b.loopDuration > 0.0f)
            ? std::fmod(lastTime_ - b.phase, b.loopDuration)
            : (lastTime_ - b.phase);
        if (age <= travelTime + 8.0f) ++n;
    }
    return n;
}

// ─── Upload + barrier ────────────────────────────────────────────────────────

void SdfRenderer::writeSlotBinding(uint32_t slot, uint32_t binding, const Buffer& buf, VkDescriptorType type) {
    DescriptorWriter writer(app_->getDevice());
    writer.writeBuffer(sdfSets[slot], binding, type, buf.buffer, 0, VK_WHOLE_SIZE);
    writer.flush();
}

void SdfRenderer::ensureSlotCapacity(uint32_t slot) {
    SdfFrameSlot& f = slots[slot];
    const size_t maxRange = static_cast<size_t>(app_->getMaxStorageBufferRange());
    auto grow = [&](Buffer& buf, uint32_t& cap, size_t elemSize, size_t need, uint32_t binding) {
        if (need < 1) need = 1; // keep bindings valid even when the scene is empty
        if (need <= cap) return;
        // Grow with headroom (mirrors DebugSDFRenderer::ensureCullCapacity);
        // clamp so VK_WHOLE_SIZE never exceeds maxStorageBufferRange.
        size_t newCap = need + need / 4 + 64;
        const size_t maxFit = maxRange / elemSize;
        if (newCap > maxFit) newCap = maxFit;
        if (newCap < need) newCap = need; // single huge scene: validation may warn, still bind fully
        if (buf.buffer != VK_NULL_HANDLE)
            app_->resources.removeBufferVma(buf.buffer, buf.allocation);
        buf = app_->createBuffer(static_cast<VkDeviceSize>(newCap * elemSize),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        cap = static_cast<uint32_t>(newCap);
        // Rewritten only on (re)allocation, never per frame, so the write
        // cannot race a pending CB in steady state.
        writeSlotBinding(slot, binding, buf, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    grow(f.instance, f.instanceCap, sizeof(SdfInstance), pendingScene_.instances().size(), 0);
    grow(f.definition, f.definitionCap, sizeof(SdfDefinition), pendingScene_.definitions().size(), 1);
    grow(f.material, f.materialCap, sizeof(SdfMaterial), pendingScene_.materials().size(), 2);
    grow(f.container, f.containerCap, sizeof(SdfContainer), pendingScene_.containers().size(), 3);
    grow(f.gridCell, f.gridCellCap, sizeof(SdfGridCell), pendingScene_.cells().size(), 4);
    grow(f.gridIndex, f.gridIndexCap, sizeof(uint32_t), pendingScene_.indices().size(), 5);
}

void SdfRenderer::flushSlotUploads(uint32_t slot) {
    if (slot >= SDF_FRAMES) return;
    SdfFrameSlot& f = slots[slot];
    if (sceneDirtySlots_[slot]) {
        ensureSlotCapacity(slot);
        auto copy = [](Buffer& dst, const void* src, size_t bytes) {
            if (bytes == 0 || dst.mappedData == nullptr || src == nullptr) return;
            std::memcpy(dst.mappedData, src, bytes);
        };
        copy(f.instance, pendingScene_.instances().data(), pendingScene_.instances().size() * sizeof(SdfInstance));
        copy(f.definition, pendingScene_.definitions().data(), pendingScene_.definitions().size() * sizeof(SdfDefinition));
        copy(f.material, pendingScene_.materials().data(), pendingScene_.materials().size() * sizeof(SdfMaterial));
        copy(f.container, pendingScene_.containers().data(), pendingScene_.containers().size() * sizeof(SdfContainer));
        copy(f.gridCell, pendingScene_.cells().data(), pendingScene_.cells().size() * sizeof(SdfGridCell));
        copy(f.gridIndex, pendingScene_.indices().data(), pendingScene_.indices().size() * sizeof(uint32_t));
        sceneDirtySlots_[slot] = false;
    }
    if (paramsDirtySlots_[slot]) {
        if (f.params.mappedData != nullptr)
            std::memcpy(f.params.mappedData, &params_, sizeof(params_));
        paramsDirtySlots_[slot] = false;
    }
}

void SdfRenderer::flushSmokeUpload(uint32_t slot) {
    if (slot >= SDF_FRAMES) return;
    SdfFrameSlot& f = slots[slot];
    if (!smokeDirtySlots_[slot]) return;
    if (f.smoke.mappedData != nullptr)
        std::memcpy(f.smoke.mappedData, &smokeState_, sizeof(smokeState_));
    smokeDirtySlots_[slot] = false;
}

void SdfRenderer::refreshDepthBinding(uint32_t slot) {
    if (depthSampler == VK_NULL_HANDLE) return;
    DescriptorWriter writer(app_->getDevice());
    bool wrote = false;
    if (pendingDepthView_ != VK_NULL_HANDLE && boundDepthViews_[slot] != pendingDepthView_) {
        writer.writeImage(sdfSets[slot], 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            depthSampler, pendingDepthView_, pendingDepthLayout_);
        boundDepthViews_[slot] = pendingDepthView_;
        wrote = true;
    }
    // Binding 9 is rewritten only when the water view handle changes (the
    // enabled flag lives in the params UBO, not the descriptor).
    if (pendingWaterDepthView_ != VK_NULL_HANDLE &&
        boundWaterDepthViews_[slot] != pendingWaterDepthView_) {
        writer.writeImage(sdfSets[slot], 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            depthSampler, pendingWaterDepthView_, pendingWaterDepthLayout_);
        boundWaterDepthViews_[slot] = pendingWaterDepthView_;
        wrote = true;
    }
    if (wrote) writer.flush();
}

void SdfRenderer::recordHostToShaderBarrier(VkCommandBuffer cmd, uint32_t slot) const {
    // HOST writes (slot SSBO + params memcpy) -> VERTEX/FRAGMENT reads.
    // Same-thread writes happen-before submit; this Sync2 barrier makes them
    // visible to the shader stages on the recording queue. NULL handles are
    // skipped (a slot whose buffers were never allocated must not emit a
    // barrier entry — VUID forbids VK_NULL_HANDLE there).
    const SdfFrameSlot& f = slots[slot];
    VkBuffer bufs[8] = {f.instance.buffer, f.definition.buffer, f.material.buffer,
                         f.container.buffer, f.gridCell.buffer, f.gridIndex.buffer, f.params.buffer,
                         f.smoke.buffer};
    VkBufferMemoryBarrier2 barriers[8]{};
    uint32_t n = 0;
    for (int i = 0; i < 8; ++i) {
        if (bufs[i] == VK_NULL_HANDLE) continue;
        barriers[n].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[n].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barriers[n].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        barriers[n].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barriers[n].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barriers[n].buffer = bufs[i];
        barriers[n].offset = 0;
        barriers[n].size = VK_WHOLE_SIZE;
        ++n;
    }
    if (n == 0) return;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = n;
    dep.pBufferMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void SdfRenderer::prepareCull(VkCommandBuffer cmd) {
    if (app_ == nullptr) return;
    const uint32_t slot = currentFrame_ % SDF_FRAMES;
    {
        std::lock_guard<std::mutex> lock(sceneMutex);
        flushSlotUploads(slot);
        flushSmokeUpload(slot);
        refreshDepthBinding(slot);
    }
    if (stats_.containerCount == 0) return;
    recordHostToShaderBarrier(cmd, slot);
}

void SdfRenderer::prepareShadowCascade(VkCommandBuffer cmd, uint32_t frameIdx) {
    if (app_ == nullptr || cmd == VK_NULL_HANDLE) return;
    setFrame(frameIdx);
    const uint32_t slot = currentFrame_ % SDF_FRAMES;
    {
        std::lock_guard<std::mutex> lock(sceneMutex);
        // Flush this frame's slot. Called from every cascade CB, but the
        // dirty flags make the host memcpy run exactly once per frame; the
        // later SDF task's prepareCull then sees clean flags and never
        // rewrites the scene buffers while a cascade CB may still read them
        // (params/smoke writes target buffers the grass shadow pipeline does
        // not read; the scene buffers are gated by sceneDirtySlots_, cleared
        // here). refreshDepthBinding is deliberately NOT called: the pending
        // depth view is the PREVIOUS frame's per-slot view here, and binding 7
        // is unused by the grass shadow shader — rewriting it would flap the
        // binding and force a second (in-use) descriptor update from the SDF
        // task. The SDF task keeps maintaining binding 7 exactly as before.
        flushSlotUploads(slot);
        flushSmokeUpload(slot);
    }
    if (stats_.containerCount == 0) return;
    // HOST writes -> shader reads, recorded BEFORE beginShadowPass because
    // buffer barriers are not allowed inside the dynamic rendering scope
    // (VUID-vkCmdPipelineBarrier2-srcStageMask-09556). Each cascade CB
    // records its own barrier (they may run on different queues).
    recordHostToShaderBarrier(cmd, slot);
}

void SdfRenderer::setShadowLightDirection(const glm::vec3& dir) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    shadowLightDir_ = dir;
}

float SdfRenderer::getTime() const {
    std::lock_guard<std::mutex> lock(sceneMutex);
    return params_.time;
}

// ─── Draw ────────────────────────────────────────────────────────────────────

void SdfRenderer::render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet mainDescriptorSet, uint32_t frameIdx, bool enabled) {
    const uint32_t slot = frameIdx % SDF_FRAMES;
    currentFrame_ = slot;

    VkImageView colorView = getSdfColorView(frameIdx);
    VkImageView depthView = getSdfDepthView(frameIdx);
    if (colorView == VK_NULL_HANDLE || depthView == VK_NULL_HANDLE) return;

    VkImage colorImg = getSdfColorImage(frameIdx);
    VkImage depthImg = getSdfDepthImage(frameIdx);
    VkImageLayout colorOld = getSdfColorLayout(frameIdx);
    VkImageLayout depthOld = getSdfDepthLayout(frameIdx);
    app->recordTransitionImageLayoutLayer(cmd, colorImg, app->getSwapchainImageFormat(), colorOld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1, 0, 1);
    setSdfColorLayout(frameIdx, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // ── Solid depth pre-load (hardware depth test against rasterized solid) ──
    // Copy this frame's rasterized solid depth into the SDF depth attachment
    // before the pass. The SDF pipeline keeps depth test/write on with
    // LESS_OR_EQUAL, so any SDF fragment whose marched hit (gl_FragDepth) is
    // behind the solid surface is rejected at raster time: grass/fire/smoke
    // cannot draw through terrain even if the in-shader tExit clamp or the
    // composite depth test were bypassed. The same solid depth is sampled
    // (set=1 binding 7) by the shader afterwards, so it is transitioned back
    // to SHADER_READ_ONLY before the draws. Fallback (no solid depth image):
    // clear the SDF depth exactly as before.
    VkImage solidDepthImage = VK_NULL_HANDLE;
    uint32_t solidDepthW = 0, solidDepthH = 0;
    {
        std::lock_guard<std::mutex> lock(sceneMutex);
        solidDepthImage = pendingDepthImage_;
        solidDepthW = pendingDepthWidth_;
        solidDepthH = pendingDepthHeight_;
    }
    const bool depthPreloaded = (solidDepthImage != VK_NULL_HANDLE && solidDepthW > 0 && solidDepthH > 0);
    if (depthPreloaded) {
        // solid: SHADER_READ_ONLY -> TRANSFER_SRC
        app->recordTransitionImageLayoutLayer(cmd, solidDepthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, 0, 1);
        // sdf: previous layout -> TRANSFER_DST
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            depthOld, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.extent = {std::min(sdfRenderWidth, solidDepthW), std::min(sdfRenderHeight, solidDepthH), 1};
        vkCmdCopyImage(cmd, solidDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       depthImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        // solid: back to SHADER_READ_ONLY for the binding-7 sampling below.
        app->recordTransitionImageLayoutLayer(cmd, solidDepthImage, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        // sdf: TRANSFER_DST -> depth attachment (loaded, not cleared).
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
    } else {
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT,
            depthOld, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
    }
    setSdfDepthLayout(frameIdx, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    VkClearValue colorClear{}; colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    if (depthPreloaded) {
        RendererUtils::beginColorDepthPassLoadDepth(cmd, colorView, depthView,
            sdfRenderWidth, sdfRenderHeight, colorClear);
    } else {
        VkClearValue depthClear{}; depthClear.depthStencil = {1.0f, 0};
        RendererUtils::beginColorDepthPass(cmd, colorView, depthView,
            sdfRenderWidth, sdfRenderHeight, colorClear, depthClear);
    }

    // Nothing to draw (disabled, no pipeline yet, empty scene): clear only, then SRO.
    const uint32_t instanceCount = stats_.containerCount;
    if (!enabled || pipeline == VK_NULL_HANDLE || instanceCount == 0 ||
        vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE || indexCount == 0) {
        vkCmdEndRendering(cmd);
        app->recordTransitionImageLayoutLayer(cmd, colorImg, app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        setSdfColorLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        setSdfDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        stats_.lastDrawInstances = 0;
        return;
    }

    SdfFrameSlot& f = slots[slot];
    if (f.instance.buffer == VK_NULL_HANDLE) {
        vkCmdEndRendering(cmd);
        app->recordTransitionImageLayoutLayer(cmd, colorImg, app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
        setSdfColorLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        setSdfDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        stats_.lastDrawInstances = 0;
        return;
    }

    // HOST -> shader barrier, repeated here even though prepareCull recorded
    // one: prepareCull's barrier lives on the cull CB, which may be a
    // different command buffer/queue; without this the draw can read
    // pre-upload state. Same pattern as DebugSDFRenderer::render.
    recordHostToShaderBarrier(cmd, slot);

    if (cmdState) cmdState->bindGraphicsPipeline(cmd, pipeline);
    else vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDescriptorSet descriptorSets[] = {mainDescriptorSet, sdfSets[slot]};
    if (cmdState) cmdState->bindGraphicsDescriptorSets(cmd, pipelineLayout, 0, 2, descriptorSets, 0, nullptr);
    else vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
        0, 2, descriptorSets, 0, nullptr);

    const VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // One proxy cube per container; the vertex shader maps [0,1]^3 onto
    // containers[gl_InstanceIndex].boundsMin/Max (instance buffer carries the
    // traversal payload for the fragment stage).
    vkCmdDrawIndexed(cmd, indexCount, instanceCount, 0, 0, 0);
    stats_.lastDrawInstances = instanceCount;
    // lastCellVisits / lastFragments stay 0 (stubs until a query pool is wired).

    vkCmdEndRendering(cmd);
    app->recordTransitionImageLayoutLayer(cmd, colorImg, app->getSwapchainImageFormat(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
    setSdfColorLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    setSdfDepthLayout(frameIdx, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void SdfRenderer::drawShadowCascade(VkCommandBuffer cmd, uint32_t cascadeIndex,
                                    const glm::mat4& lightViewProj, float time) {
    (void)cascadeIndex; // one draw serves every cascade; only the matrix changes
    if (app_ == nullptr || cmd == VK_NULL_HANDLE) return;
    if (shadowPipeline == VK_NULL_HANDLE || shadowPipelineLayout == VK_NULL_HANDLE) return;
    const uint32_t slot = currentFrame_ % SDF_FRAMES;
    if (slots[slot].container.buffer == VK_NULL_HANDLE || sdfSets[slot] == VK_NULL_HANDLE) return;
    if (vertexBuffer.buffer == VK_NULL_HANDLE || indexBuffer.buffer == VK_NULL_HANDLE ||
        indexCount == 0) return;

    uint32_t grassBase = 0;
    uint32_t grassCount = 0;
    SdfGrassShadowPC pc{};
    {
        std::lock_guard<std::mutex> lock(sceneMutex);
        // Grass-only gate: the pass rasterizes the GRASS container range only
        // (computed by refreshMergedLocked from the merge order), so fire,
        // smoke and rock containers are never drawn; the fragment shader
        // additionally skips every non-grass definition before evaluation.
        if (!config_.grass.enabled) return;
        grassBase = grassContainerBase_;
        grassCount = grassContainerCount_;
        if (grassCount == 0) return;
        pc.lightViewProj = lightViewProj;
        pc.params = glm::vec4(time, params_.maxSteps, params_.epsilon, params_.safety);
        // march.z = shadow LOD camera scale: the grass shadow march evaluates
        // the reduced blade set (config_.grass.shadowLodScale) instead of the
        // full clump, per the shadow LOD policy; >= 80 (SdfGrass.glsl
        // GRASS_IMPOSTOR_FULL) it evaluates the impostor only (zero blades).
        pc.march = glm::vec4(params_.maxStep, params_.minStep,
                             config_.grass.shadowLodScale, 0.0f);
        // Same impostor band as the main pass so shadow LOD matches the
        // rendered grass representation (streamed, no rebuild).
        pc.impostor = glm::vec4(config_.grass.impostorStart,
                                config_.grass.impostorFull, 0.0f, 0.0f);
        pc.lightDir = glm::vec4(shadowLightDir_, 0.0f);
    }

    // HOST -> shader visibility: prepareShadowCascade recorded the barrier on
    // this command buffer BEFORE beginShadowPass (buffer barriers are not
    // allowed inside the dynamic rendering scope); same queue + same CB, so
    // the upload is ordered before this draw.
    // Raw binds (not cmdState): the SDF renderer's state tracker belongs to
    // the SDF/main command buffer, never to this cascade CB, so eliding a
    // bind through it could skip a required bind on this command buffer.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
    VkDescriptorSet set = sdfSets[slot];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout,
        1, 1, &set, 0, nullptr);

    const VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdPushConstants(cmd, shadowPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pc), &pc);

    // One proxy cube per GRASS container; firstInstance offsets into the
    // merged container array (gl_InstanceIndex includes firstInstance, so the
    // vertex shader indexes the right container).
    vkCmdDrawIndexed(cmd, indexCount, grassCount, 0, 0, grassBase);
}

// ─── Targets / lifecycle ─────────────────────────────────────────────────────

void SdfRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (sdfRenderWidth == width && sdfRenderHeight == height && sdfColorImages[0] != VK_NULL_HANDLE) {
        return; // Already created at this size
    }

    destroyRenderTargets(app);

    sdfRenderWidth = width;
    sdfRenderHeight = height;
    {
        // Ray-cast quality needs gl_FragCoord -> UV: stream the target's
        // inverse size with the other params (write-on-change via the dirty
        // flags; resize is the only writer besides init).
        std::lock_guard<std::mutex> lock(sceneMutex);
        params_.invScreenSize = glm::vec2(
            width > 0 ? 1.0f / static_cast<float>(width) : 0.0f,
            height > 0 ? 1.0f / static_cast<float>(height) : 0.0f);
        paramsDirtySlots_.fill(true);
    }

    VkDevice device = app->getDevice();

    auto createImage = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                           VkImage& image, VmaAllocation& allocation, VkDeviceMemory& memory, VkImageView& view) {
        RendererUtils::createImage2DWithVma(device, app, width, height, format, usage, aspect,
                                            "SdfRenderer: offscreen", image, allocation, memory, view);
    };

    for (uint32_t i = 0; i < SDF_FRAMES; ++i) {
        createImage(app->getSwapchainImageFormat(),
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    sdfColorImages[i], sdfColorAllocations[i], sdfColorMemories[i], sdfColorImageViews[i]);
        if (sdfColorImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(sdfColorImages[i], app->getSwapchainImageFormat(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(sdfColorImages[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1);
            sdfColorImageLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        createImage(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    sdfDepthImages[i], sdfDepthAllocations[i], sdfDepthMemories[i], sdfDepthImageViews[i]);
        if (sdfDepthImages[i] != VK_NULL_HANDLE && app) {
            app->transitionImageLayoutLayer(sdfDepthImages[i], VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
            app->setImageLayoutTracked(sdfDepthImages[i], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, 1);
            sdfDepthImageLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        // Binding 7 placeholder: sample our own depth until setSceneDepth()
        // provides the main scene depth. Written once here (not per frame).
        boundDepthViews_[i] = VK_NULL_HANDLE;
        if (sdfDepthImageViews[i] != VK_NULL_HANDLE && depthSampler != VK_NULL_HANDLE &&
            sdfSets[i] != VK_NULL_HANDLE) {
            DescriptorWriter writer(app->getDevice());
            writer.writeImage(sdfSets[i], 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                depthSampler, sdfDepthImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            // Binding 9 (water depth) placeholder: same self-depth view so the
            // set is complete before the first setWaterDepth(); the shader
            // only samples it while params_.waterDepthEnabled > 0.5, so the
            // placeholder is never read as water.
            writer.writeImage(sdfSets[i], 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                depthSampler, sdfDepthImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writer.flush();
            boundDepthViews_[i] = sdfDepthImageViews[i];
            boundWaterDepthViews_[i] = sdfDepthImageViews[i];
            // A later setSceneDepth() with a different view replaces this via refreshDepthBinding().
            if (pendingDepthView_ != VK_NULL_HANDLE && pendingDepthView_ != sdfDepthImageViews[i])
                boundDepthViews_[i] = VK_NULL_HANDLE; // force refresh to the pending view
            if (pendingWaterDepthView_ != VK_NULL_HANDLE &&
                pendingWaterDepthView_ != sdfDepthImageViews[i])
                boundWaterDepthViews_[i] = VK_NULL_HANDLE;
        }
    }

    std::cout << "[SdfRenderer] Created offscreen render targets " << width << "x" << height << std::endl;
}

void SdfRenderer::destroyRenderTargets(VulkanApp* app) {
    (void)app;
    for (uint32_t i = 0; i < SDF_FRAMES; ++i) {
        sdfColorImages[i] = VK_NULL_HANDLE;
        sdfColorAllocations[i] = VK_NULL_HANDLE;
        sdfColorMemories[i] = VK_NULL_HANDLE;
        sdfColorImageViews[i] = VK_NULL_HANDLE;
        sdfDepthImages[i] = VK_NULL_HANDLE;
        sdfDepthAllocations[i] = VK_NULL_HANDLE;
        sdfDepthMemories[i] = VK_NULL_HANDLE;
        sdfDepthImageViews[i] = VK_NULL_HANDLE;
        boundDepthViews_[i] = VK_NULL_HANDLE;
        boundWaterDepthViews_[i] = VK_NULL_HANDLE;
    }
}

void SdfRenderer::onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height) {
    // Call only when no frame in flight references the old targets (same
    // constraint as every other offscreen renderer). Rebuilds at the new size.
    createRenderTargets(app, width, height);
}

void SdfRenderer::cleanup(VulkanApp* app) {
    (void)app;
    vertexBuffer = {};
    indexBuffer = {};
    indexCount = 0;
    for (auto& s : slots) {
        s.instance = {};
        s.definition = {};
        s.material = {};
        s.container = {};
        s.gridCell = {};
        s.gridIndex = {};
        s.params = {};
        s.smoke = {};
        s.instanceCap = s.definitionCap = s.materialCap = 0;
        s.containerCap = s.gridCellCap = s.gridIndexCap = 0;
    }
    depthSampler = VK_NULL_HANDLE;
    stats_ = SdfStats{};
    app_ = nullptr;
}
