#include "SdfRenderer.hpp"
#include "sdf/types/SdfProxyVertex.hpp"
#include "vulkan/renderer/SdfSceneFlatten.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorWriter.hpp"
#include "RendererUtils.hpp"
#include "../ShaderStage.hpp"
#include "../includes/locations.hpp"
#include "../../math/Geometry.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

SdfRenderer::SdfRenderer() = default;

SdfRenderer::~SdfRenderer() { cleanup(nullptr); }

void SdfRenderer::init(VulkanApp* app) {
    app_ = app;
    createCubeBuffers(app);
    createDescriptorSet(app);
    depthSampler = app->createSamplerLinearClamp("SdfRenderer: sceneDepthSampler");
    // Params UBOs: one per slot, created once (fixed size, never regrown).
    // SdfParamsUBO layout: timeDebug=(time,packedMode,maxSteps,safety),
    // marchParams=(minStep,maxStep,epsilon,earlyTerm). Defaults match the
    // fragment shader fallbacks so a zero time still raymarches sanely.
    params_.timeDebug = glm::vec4(0.0f, 0.0f, 64.0f, 0.7f);
    params_.marchParams = glm::vec4(0.05f, std::clamp(config_.lava.scale, 1.0f, 32.0f), 0.01f, 0.99f);
    params_.fireColors0 = glm::vec4(0.6f, 0.05f, 0.0f, 0.0f);
    params_.fireColors1 = glm::vec4(1.0f, 0.95f, 0.6f, 0.0f);
    repackDebugMode();
    renderMode_ = RenderMode::Volume; // fire-first default; generic modes via setRenderMode
    repackDebugMode();
    for (uint32_t s = 0; s < SDF_FRAMES; ++s) {
        SdfFrameSlot& slot = slots[s];
        slot.params = app->createBuffer(sizeof(SdfParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        writeSlotBinding(s, 6, slot.params, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        // Smoke state: fixed size, created once, streamed on demand.
        slot.smoke = app->createBuffer(sizeof(SmokeFragBulletGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        writeSlotBinding(s, 8, slot.smoke, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    sceneDirtySlots_.fill(true);
    paramsDirtySlots_.fill(true);
    smokeDirtySlots_.fill(true);
    // Default smoke tuning mirrors SdfEffectConfig defaults (single source:
    // SdfSmokeConfig for the cloud, SdfBulletConfig for bullet FX/gold); the
    // SSBO mirror is packed here once, widget edits stream through the
    // setters below.
    smokeState_.tuning.timing = glm::vec4(config_.smoke.growthDuration,
        config_.smoke.loopDuration, config_.smoke.dissipation, 0.0f);
    smokeState_.tuning.noise = glm::vec4(config_.smoke.noiseScale,
        config_.smoke.noiseStrength, config_.smoke.noiseWarp, 0.0f);
    {
        const float rad = glm::radians(config_.smoke.windAngleDeg);
        smokeState_.tuning.wind = glm::vec4(config_.smoke.windSpeed * std::cos(rad),
            config_.smoke.windSpeed * std::sin(rad), config_.smoke.densityScale, 0.0f);
    }
    smokeState_.tuning.tunnel = glm::vec4(config_.bullet.tunnelStrength,
        config_.bullet.tunnelFalloff, config_.bullet.wakeStrength, config_.bullet.wakeDissipation);
    smokeState_.tuning.wake = glm::vec4(config_.bullet.wakeRadius,
        config_.bullet.wakeExpansion, config_.bullet.wakeLength, 0.0f);
    smokeState_.tuning.pressure = glm::vec4(config_.bullet.pressureRadius,
        config_.bullet.pressureStrength, config_.bullet.pressureWaveSpeed,
        config_.bullet.pressureWaveFreq);
    smokeState_.tuning.turbWave = glm::vec4(config_.bullet.pressureWaveFalloff,
        config_.bullet.turbScale, config_.bullet.turbStrength, config_.bullet.turbSpeed);
    smokeState_.tuning.render = glm::vec4(static_cast<float>(config_.smoke.shadowSamples),
        config_.smoke.shadowStrength, 0.0f, 0.0f);
    smokeState_.tuning.gold0 = glm::vec4(config_.bullet.goldDeep, config_.bullet.goldSpecPower);
    smokeState_.tuning.gold1 = glm::vec4(config_.bullet.goldBright, config_.bullet.goldPatternScale);
    smokeState_.tuning.gold2 = glm::vec4(config_.bullet.goldSpecStrength,
        config_.bullet.goldFresnelBoost, config_.bullet.goldWarmFloor, config_.bullet.goldNormalDistort);
    refreshAutoBulletLocked();
    ensureSmokeScene();
    refreshMergedLocked();
    createPipeline(app);
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

    // set=1 bindings 0..8: 0 instances, 1 definitions, 2 materials,
    // 3 containers, 4 gridCells, 5 gridIndices, 6 params UBO, 7 sceneDepth,
    // 8 smoke state (tuning + bullets, fragment only).
    VkDescriptorSetLayoutBinding bindings[9]{};
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

    descriptorSetLayout = descAlloc.createLayout(
        bindings, 9, 0, nullptr, "SdfRenderer: descriptorSetLayout");

    VkDescriptorPoolSize poolSizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * SDF_FRAMES},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * SDF_FRAMES},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 * SDF_FRAMES},
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
        vertModule = app->getOrCreateShaderModule("shaders/sdf.vert.spv");
        fragModule = app->getOrCreateShaderModule("shaders/sdf.frag.spv");
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

// ─── Scene input ─────────────────────────────────────────────────────────────

void SdfRenderer::repackDebugMode() {
    const uint32_t packed = (static_cast<uint32_t>(renderMode_) << 16) | (debugFlags_ & 0xFFFFu);
    params_.timeDebug.y = static_cast<float>(packed); // exact: packed < 2^24
}

void SdfRenderer::extractFlattened() {
    // Single FlattenSdfScene() call keeps container gridInfo/gridOffset
    // consistent with the global cell/index arrays (see SdfSceneFlatten.hpp).
    SdfFlattenedScene flat = FlattenSdfScene(pendingScene_);
    instances_ = std::move(flat.instances);
    definitions_ = std::move(flat.definitions);
    materials_ = std::move(flat.materials);
    containers_ = std::move(flat.containers);
    gridCells_ = std::move(flat.cells);
    gridIndices_ = std::move(flat.indices);

    stats_.containerCount = static_cast<uint32_t>(containers_.size());
    stats_.definitionCount = static_cast<uint32_t>(definitions_.size());
    stats_.materialCount = static_cast<uint32_t>(materials_.size());
    stats_.gridCellCount = static_cast<uint32_t>(gridCells_.size());
}

void SdfRenderer::setScene(const sdf_gpu::SdfScene& scene) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    lavaScene_ = scene;
    refreshMergedLocked();
}

void SdfRenderer::refreshMergedLocked() {
    // pendingScene_ is always the merge: independent emitters (lava fire,
    // smoke bombs, future effects) share one GPU upload and one march.
    pendingScene_ = sdf_gpu::SdfScene::merge(lavaScene_, smokeScene_);
    extractFlattened();
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
            const glm::vec3 mn = smokeScene_.containers().front().aabb.getMin();
            const glm::vec3 mx = smokeScene_.containers().front().aabb.getMax();
            fprintf(stderr, "[SdfRenderer] scene: smoke instances=%zu "
                "container=[(%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f)] lava instances=%zu\n",
                smokeCount, mn.x, mn.y, mn.z,
                mx.x, mx.y, mx.z, lavaCount);
        } else {
            fprintf(stderr, "[SdfRenderer] scene: smoke DISABLED/empty, lava instances=%zu\n",
                lavaCount);
        }
        const SmokeGPU& st = smokeState_.tuning;
        const BulletGPU& b0 = smokeState_.bullets[0];
        float mden = -1.0f, mabs = -1.0f;
        if (!smokeScene_.materials().empty()) {
            mden = smokeScene_.materials()[0].density;
            mabs = smokeScene_.materials()[0].absorption;
        }
        fprintf(stderr, "[SdfRenderer] smoke state: timing=(%.2f,%.2f,%.2f) "
            "noise=(%.4f,%.2f,%.2f) wind=(%.2f,%.2f,%.2f) mat(d=%.2f,a=%.2f) "
            "march(maxStep=%.2f,maxSteps=%.0f) mode=%u flags=%u\n",
            st.timing.x, st.timing.y, st.timing.z,
            st.noise.x, st.noise.y, st.noise.z,
            st.wind.x, st.wind.y, st.wind.z, mden, mabs,
            params_.marchParams.y, params_.timeDebug.z,
            static_cast<uint32_t>(renderMode_), debugFlags_);
        fprintf(stderr, "[SdfRenderer] bullet0: a=(%.0f,%.0f,%.0f,r%.1f) "
            "vel=(%.0f,%.0f,%.0f) len=%.0f c=(rEnd%.1f,loop%.1f,int%.1f,ph%.1f)\n",
            b0.a.x, b0.a.y, b0.a.z, b0.a.w,
            b0.b.x, b0.b.y, b0.b.z, b0.b.w,
            b0.c.x, b0.c.y, b0.c.z, b0.c.w);
    }
}

void SdfRenderer::ensureSmokeScene() {
    // Static topology: 1 def/mat/container/instance. All behavior streams
    // through the smoke SSBO, so widget tweaks below never come here —
    // only position/radius/base material reshape the scene.
    if (config_.smoke.enabled) {
        smokeScene_ = sdf_gpu::SdfScene::createSmokeBomb(config_.smoke.pos, config_.smoke.radius, config_.smoke.seed);
        // Cover bullet flight paths: bullets fly horizontally (XZ) from
        // auto-crossing starts through and past the smoke. The static margin
        // fits the longest paths; cells outside the smoke AABB stay empty
        // and march at DDA-skip cost only.
        if (!smokeScene_.containers().empty()) {
            auto& c = smokeScene_.containers()[0];
            const float ext = 1200.0f;
            const glm::vec3 mn = c.aabb.getMin() - glm::vec3(ext, 0.0f, ext);
            const glm::vec3 mx = c.aabb.getMax() + glm::vec3(ext, 0.0f, ext);
            c.aabb = BoundingBox(mn, mx);
        }
    } else {
        smokeScene_.clear();
    }
    if (!smokeScene_.materials().empty()) {
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
    params_.timeDebug.x = timeSec;
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
    params_.timeDebug.z = maxSteps;
    params_.marchParams.z = epsilon;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setMarchRange(float minStep, float maxStep, float earlyTermThreshold) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    params_.marchParams.x = minStep;
    params_.marchParams.y = maxStep;
    params_.marchParams.w = earlyTermThreshold;
    paramsDirtySlots_.fill(true);
}

void SdfRenderer::setSceneDepth(VkImageView view, VkImageLayout layout) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    pendingDepthView_ = view;
    pendingDepthLayout_ = layout;
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
    refreshMergedLocked();
    stats_.lavaAnchors = static_cast<uint32_t>(all.size());
    stats_.lavaChunks = static_cast<uint32_t>(lavaByChunk_.size());
    {
        const auto& cs = lavaScene_.containers();
        if (cs.empty()) {
            fprintf(stderr, "[SdfRenderer] lava rebuild: anchors=%zu chunks=%zu (no lava containers)\n",
                all.size(), lavaByChunk_.size());
        } else {
            const glm::vec3 mn = cs.front().aabb.getMin();
            const glm::vec3 mx = cs.front().aabb.getMax();
            fprintf(stderr, "[SdfRenderer] lava rebuild: anchors=%zu chunks=%zu "
                "container=[(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)] grid=%ux%ux%u\n",
                all.size(), lavaByChunk_.size(),
                mn.x, mn.y, mn.z, mx.x, mx.y, mx.z,
                cs.front().resolution.x, cs.front().resolution.y, cs.front().resolution.z);
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
    params_.marchParams.y = std::clamp(config_.lava.scale, 1.0f, 32.0f);
    paramsDirtySlots_.fill(true);
}


void SdfRenderer::setLavaSpikiness(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(s, 0.0f, 1.5f);
    if (v == config_.lava.spikiness) return;
    config_.lava.spikiness = v;
    lavaDirty_ = true; // def params changed -> re-flatten on next rebuild
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


// ─── Smoke bomb + bullets ─────────────────────────────────────────────────
// Scene-affecting setters rebuild the static smoke topology (cheap: 1 def /
// mat / container / instance) and re-merge; tuning setters only stream the
// smoke SSBO (no re-flatten).

void SdfRenderer::markSmokeSSBO() {
    smokeDirtySlots_.fill(true);
}

void SdfRenderer::refreshAutoBulletLocked() {
    const float rad = glm::radians(config_.bullet.angleDeg);
    const glm::vec3 dir(std::cos(rad), 0.0f, std::sin(rad));
    const glm::vec3 start = config_.smoke.pos - dir * (config_.smoke.radius + 60.0f);
    // BulletGPU ABI: a = (start, radiusStart), b = (velocity m/s, length),
    // c = (radiusEnd, loopDuration, intensity, phase). Auto bullet: phase =
    // growthDuration so it waits out the smoke bloom inside each loop; the
    // auto-fire flag lives in c.z (0 = empty slot).
    BulletGPU& b = smokeState_.bullets[0];
    b.a = glm::vec4(start, config_.bullet.radius);
    b.b = glm::vec4(dir * config_.bullet.speed, config_.bullet.length);
    b.c = glm::vec4(config_.bullet.finalRadius, config_.bullet.loopDuration,
                    1.0f, config_.smoke.growthDuration);
    syncAutoBulletLocked(); // applies autoFire + single-flight (marks dirty on change)
    markSmokeSSBO();
}

bool SdfRenderer::anyManualBulletLiveLocked() const {
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i) {
        const BulletGPU& b = smokeState_.bullets[i];
        if (b.c.z <= 0.0f) continue;
        const float speed = std::max(glm::length(glm::vec3(b.b)), 1e-3f);
        const float travelTime = std::max(b.b.w, 1e-3f) / speed;
        const float age = (b.c.y > 0.0f)
            ? std::fmod(lastTime_ - b.c.w, b.c.y)
            : (lastTime_ - b.c.w);
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
    if (smokeState_.bullets[0].c.z != want) {
        smokeState_.bullets[0].c.z = want;
        markSmokeSSBO();
    }
}

void SdfRenderer::setSmokeEnabled(bool e) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    if (e == config_.smoke.enabled) return;
    config_.smoke.enabled = e;
    ensureSmokeScene();
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


void SdfRenderer::setSmokeRadius(float r) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    const float v = std::clamp(r, 8.0f, 1024.0f);
    if (v == config_.smoke.radius) return;
    config_.smoke.radius = v;
    ensureSmokeScene();
    refreshAutoBulletLocked();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeGrowthDuration(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.growthDuration = std::clamp(s, 0.5f, 60.0f);
    smokeState_.tuning.timing.x = config_.smoke.growthDuration;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeLoopDuration(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.loopDuration = std::clamp(s, 2.0f, 120.0f);
    smokeState_.tuning.timing.y = config_.smoke.loopDuration;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeDissipation(float d) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.dissipation = std::clamp(d, 0.0f, 2.0f);
    smokeState_.tuning.timing.z = config_.smoke.dissipation;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseScale(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseScale = std::clamp(s, 0.001f, 0.2f);
    smokeState_.tuning.noise.x = config_.smoke.noiseScale;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseStrength(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseStrength = std::clamp(s, 0.0f, 2.0f);
    smokeState_.tuning.noise.y = config_.smoke.noiseStrength;
    markSmokeSSBO();
}


void SdfRenderer::setSmokeNoiseWarp(float s) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.noiseWarp = std::clamp(s, 0.0f, 2.0f);
    smokeState_.tuning.noise.z = config_.smoke.noiseWarp;
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
    smokeState_.tuning.wind.z = config_.smoke.densityScale;
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
    const float v = std::clamp(s, 0.0f, 2.0f);
    if (v == config_.smoke.scattering) return;
    config_.smoke.scattering = v;
    ensureSmokeScene();
    refreshMergedLocked();
}


void SdfRenderer::setSmokeTunnel(float strength, float falloff) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.tunnelStrength = std::clamp(strength, 0.0f, 1.0f);
    smokeState_.tuning.tunnel.x = config_.bullet.tunnelStrength;
    config_.bullet.tunnelFalloff = std::clamp(falloff, 0.5f, 100.0f);
    smokeState_.tuning.tunnel.y = config_.bullet.tunnelFalloff;
    markSmokeSSBO();
}



void SdfRenderer::setSmokeWake(float strength, float radius, float expansion, float length, float dissipation) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.wakeStrength = std::clamp(strength, 0.0f, 2.0f);
    smokeState_.tuning.tunnel.z = config_.bullet.wakeStrength;
    config_.bullet.wakeRadius = std::clamp(radius, 0.5f, 100.0f);
    smokeState_.tuning.wake.x = config_.bullet.wakeRadius;
    config_.bullet.wakeExpansion = std::clamp(expansion, 0.0f, 1.0f);
    smokeState_.tuning.wake.y = config_.bullet.wakeExpansion;
    config_.bullet.wakeLength = std::clamp(length, 1.0f, 500.0f);
    smokeState_.tuning.wake.z = config_.bullet.wakeLength;
    config_.bullet.wakeDissipation = std::clamp(dissipation, 0.05f, 3.0f);
    smokeState_.tuning.tunnel.w = config_.bullet.wakeDissipation;
    markSmokeSSBO();
}






void SdfRenderer::setSmokePressure(float radius, float strength, float waveSpeed, float waveFreq, float waveFalloff) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.pressureRadius = std::clamp(radius, 0.5f, 150.0f);
    smokeState_.tuning.pressure.x = config_.bullet.pressureRadius;
    config_.bullet.pressureStrength = std::clamp(strength, 0.0f, 30.0f);
    smokeState_.tuning.pressure.y = config_.bullet.pressureStrength;
    config_.bullet.pressureWaveSpeed = std::clamp(waveSpeed, 0.0f, 300.0f);
    smokeState_.tuning.pressure.z = config_.bullet.pressureWaveSpeed;
    config_.bullet.pressureWaveFreq = std::clamp(waveFreq, 0.01f, 3.0f);
    smokeState_.tuning.pressure.w = config_.bullet.pressureWaveFreq;
    config_.bullet.pressureWaveFalloff = std::clamp(waveFalloff, 0.001f, 1.0f);
    smokeState_.tuning.turbWave.x = config_.bullet.pressureWaveFalloff;
    markSmokeSSBO();
}






void SdfRenderer::setSmokeTurbulence(float scale, float strength, float speed) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.turbScale = std::clamp(scale, 0.001f, 1.0f);
    smokeState_.tuning.turbWave.y = config_.bullet.turbScale;
    config_.bullet.turbStrength = std::clamp(strength, 0.0f, 10.0f);
    smokeState_.tuning.turbWave.z = config_.bullet.turbStrength;
    config_.bullet.turbSpeed = std::clamp(speed, 0.0f, 10.0f);
    smokeState_.tuning.turbWave.w = config_.bullet.turbSpeed;
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
    smokeState_.tuning.gold0 = glm::vec4(config_.bullet.goldDeep, config_.bullet.goldSpecPower);
    smokeState_.tuning.gold1 = glm::vec4(config_.bullet.goldBright, config_.bullet.goldPatternScale);
    smokeState_.tuning.gold2 = glm::vec4(config_.bullet.goldSpecStrength,
        config_.bullet.goldFresnelBoost, config_.bullet.goldWarmFloor, config_.bullet.goldNormalDistort);
    markSmokeSSBO();
}




void SdfRenderer::setSmokeShadow(int samples, float strength) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.smoke.shadowSamples = std::clamp(samples, 1, 8);
    smokeState_.tuning.render.x = static_cast<float>(config_.smoke.shadowSamples);
    config_.smoke.shadowStrength = std::clamp(strength, 0.0f, 1.0f);
    smokeState_.tuning.render.y = config_.smoke.shadowStrength;
    markSmokeSSBO();
}



void SdfRenderer::setBulletDefaults(float radiusStart, float radiusEnd, float speed,
                                    float length, float angleDeg, float loopDuration) {
    std::lock_guard<std::mutex> lock(sceneMutex);
    config_.bullet.radius = std::clamp(radiusStart, 0.5f, 256.0f);
    config_.bullet.finalRadius = std::clamp(radiusEnd, 0.5f, 256.0f);
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
    const glm::vec3 dir(std::cos(rad), 0.0f, std::sin(rad));
    const glm::vec3 start = config_.smoke.pos - dir * (config_.smoke.radius + 60.0f);
    // Single-flight: exactly one bullet at a time. Clear any previous manual
    // round (rapid firing can't stack tunnels) and stamp the single manual
    // slot; slot 0 stays the auto-loop template, suppressed below while the
    // manual round lives.
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i)
        smokeState_.bullets[i].c.z = 0.0f;
    BulletGPU& m = smokeState_.bullets[1];
    m.a = glm::vec4(start, config_.bullet.radius);
    // BulletGPU ABI: b = (velocity m/s, length). One-shot: c.y = 0 disables
    // the per-bullet loop in the shader, so the round flies once (born now:
    // phase = lastTime_) and then disappears; c.z = 1 marks it in flight.
    m.b = glm::vec4(dir * config_.bullet.speed, config_.bullet.length);
    m.c = glm::vec4(config_.bullet.finalRadius, 0.0f, 1.0f, lastTime_);
    syncAutoBulletLocked();
    markSmokeSSBO();
}

void SdfRenderer::clearBullets() {
    std::lock_guard<std::mutex> lock(sceneMutex);
    for (uint32_t i = 1; i < kSmokeMaxBullets; ++i) {
        smokeState_.bullets[i].c.z = 0.0f; // intensity 0 = empty
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
        const BulletGPU& b = smokeState_.bullets[i];
        if (b.c.z <= 0.0f) continue;
        const float speed = std::max(glm::length(glm::vec3(b.b)), 1e-3f);
        const float pathLen = std::max(b.b.w, 1e-3f);
        const float travelTime = pathLen / speed;
        const float age = (b.c.y > 0.0f)
            ? std::fmod(lastTime_ - b.c.w, b.c.y)
            : (lastTime_ - b.c.w);
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
    grow(f.instance, f.instanceCap, sizeof(SdfInstanceGPU), instances_.size(), 0);
    grow(f.definition, f.definitionCap, sizeof(SdfDefinitionGPU), definitions_.size(), 1);
    grow(f.material, f.materialCap, sizeof(SdfMaterialGPU), materials_.size(), 2);
    grow(f.container, f.containerCap, sizeof(SdfContainerGPU), containers_.size(), 3);
    grow(f.gridCell, f.gridCellCap, sizeof(SdfGridCellGPU), gridCells_.size(), 4);
    grow(f.gridIndex, f.gridIndexCap, sizeof(uint32_t), gridIndices_.size(), 5);
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
        copy(f.instance, instances_.data(), instances_.size() * sizeof(SdfInstanceGPU));
        copy(f.definition, definitions_.data(), definitions_.size() * sizeof(SdfDefinitionGPU));
        copy(f.material, materials_.data(), materials_.size() * sizeof(SdfMaterialGPU));
        copy(f.container, containers_.data(), containers_.size() * sizeof(SdfContainerGPU));
        copy(f.gridCell, gridCells_.data(), gridCells_.size() * sizeof(SdfGridCellGPU));
        copy(f.gridIndex, gridIndices_.data(), gridIndices_.size() * sizeof(uint32_t));
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
    if (pendingDepthView_ == VK_NULL_HANDLE) return;
    if (boundDepthViews_[slot] == pendingDepthView_) return; // dedupe: no per-frame rewrites
    if (depthSampler == VK_NULL_HANDLE) return;
    DescriptorWriter writer(app_->getDevice());
    writer.writeImage(sdfSets[slot], 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        depthSampler, pendingDepthView_, pendingDepthLayout_);
    writer.flush();
    boundDepthViews_[slot] = pendingDepthView_;
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

    // HOST writes (slot SSBO + params memcpy above) -> VERTEX/FRAGMENT reads.
    // Same-thread writes happen-before submit; this Sync2 barrier makes them
    // visible to the shader stages on the recording queue. NULL handles are
    // skipped (a slot whose buffers were never allocated must not emit a
    // barrier entry — VUID forbids VK_NULL_HANDLE there).
    SdfFrameSlot& f = slots[slot];
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
    app->recordTransitionImageLayoutLayer(cmd, depthImg, VK_FORMAT_D32_SFLOAT, depthOld, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1, 0, 1);
    setSdfColorLayout(frameIdx, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    setSdfDepthLayout(frameIdx, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    VkClearValue colorClear{}; colorClear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkClearValue depthClear{}; depthClear.depthStencil = {1.0f, 0};
    RendererUtils::beginColorDepthPass(cmd, colorView, depthView, sdfRenderWidth, sdfRenderHeight, colorClear, depthClear);

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
    // NULL handles are skipped (see prepareCull).
    VkBuffer bufs[8] = {f.instance.buffer, f.definition.buffer, f.material.buffer,
                         f.container.buffer, f.gridCell.buffer, f.gridIndex.buffer, f.params.buffer,
                         f.smoke.buffer};
    VkBufferMemoryBarrier2 barriers[8]{};
    uint32_t nBar = 0;
    for (int i = 0; i < 8; ++i) {
        if (bufs[i] == VK_NULL_HANDLE) continue;
        barriers[nBar].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[nBar].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barriers[nBar].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        barriers[nBar].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barriers[nBar].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barriers[nBar].buffer = bufs[i];
        barriers[nBar].offset = 0;
        barriers[nBar].size = VK_WHOLE_SIZE;
        ++nBar;
    }
    if (nBar > 0) {
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = nBar;
        dep.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

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

// ─── Targets / lifecycle ─────────────────────────────────────────────────────

void SdfRenderer::createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height) {
    if (sdfRenderWidth == width && sdfRenderHeight == height && sdfColorImages[0] != VK_NULL_HANDLE) {
        return; // Already created at this size
    }

    destroyRenderTargets(app);

    sdfRenderWidth = width;
    sdfRenderHeight = height;

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
            writer.flush();
            boundDepthViews_[i] = sdfDepthImageViews[i];
            // A later setSceneDepth() with a different view replaces this via refreshDepthBinding().
            if (pendingDepthView_ != VK_NULL_HANDLE && pendingDepthView_ != sdfDepthImageViews[i])
                boundDepthViews_[i] = VK_NULL_HANDLE; // force refresh to the pending view
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
