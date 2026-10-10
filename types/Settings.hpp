#pragma once

class Settings {
public:
    // Grass representation selector (Settings widget: "Grass"):
    //   Vegetation   = legacy textured billboard/impostor draws
    //   GrassRaycast = SDF grass rendered by the generic SDF renderer
    //   None         = no grass
    enum class GrassMode { Vegetation = 0, GrassRaycast = 1, None = 2 };

    void resetToDefaults() {
        *this = Settings{};
    }

    // Global toggles
    bool enableShadows = true;
    // Volumetric clouds (sky raymarch + ground shadows + reflections).
    // Minimal preset disables; Maximum enables.
    bool cloudsEnabled = true;
    // Toggle rendering of the main solid scene (terrain/meshes)
    bool renderSolid = true;
    bool waterEnabled = true;
    // Global gate for the per-material refraction/tint blur (WaterSettings::
    // enableBlur): the final-pass blur runs only when BOTH this and the
    // layer's own flag are on (same two-way pattern as rtRefractions /
    // rtWaterReflections with the per-layer reflection/refraction toggles).
    bool blurEnabled = true;
    // Legacy billboard/impostor vegetation. Derived from grassMode every
    // frame (MyApp): true only in GrassMode::Vegetation. Grass clumps for the
    // SDF renderer stream from the same vegetation generation regardless.
    bool vegetationEnabled = false;
    // Active grass representation (see GrassMode). Default: SDF raycast.
    GrassMode grassMode = GrassMode::GrassRaycast;
    // Ray-cast quality: the SDF raymarcher AND the sky cloud raymarcher cast
    // one ray per NxN screen-pixel block (1 = one ray per pixel).
    // 2 = the default 2x2 pixelation.
    int raycastPixelSize = 2;
    // ── SDF march quality tiers (perf report 25 M12) ─────────────────────
    // Tier precedent: report-23 C1 (Settings::textureArraySize, default 1024,
    // low 512, idle-guarded reallocate) and report-24 H8 (tiered equirect
    // bake, recreate on tier change). Unlike those, march tiers need no
    // device idle and no target recreation: they stream through the existing
    // SdfRenderer setters into SdfParamsUBO (per-frame calls in MyApp, the
    // renderer dedupes unchanged values), so preset switches stay instant.
    // Maximum keeps the reference budgets; Minimal marches cheaper.
    //   sdfMaxSteps: march step budget (clamped 16..256 in the setter; the
    //     shader hard-caps at SDF_MAX_STEPS_HARD = 256). Minimal 32.
    //   sdfSmokeSamples: phase-B smoke resolve samples (clamped 4..12; the
    //     shader loop keeps the constant trip count 12 with an early break).
    //     Minimal 6. The fixed shadow-resolve constant is unchanged.
    // CPU distance-scaled maxSteps per container is NOT implemented: render()
    // issues one instanced draw over all containers from a single global UBO
    // budget, so per-container budgets need a draw split + shader plumbing
    // (see the TODO at SdfRenderer::render). Tier-only for now.
    int sdfMaxSteps = 64;
    int sdfSmokeSamples = 12;
    // Ray marching master gate: the generic SDF raymarcher (fire, smoke,
    // rocks, grass) and its grass shadow caster. The sky cloud raymarch has
    // its own Volumetric Clouds toggle. Minimal preset turns this off.
    bool rayMarchingEnabled = true;
    bool wireframeMode = false;
    bool waterWireframeMode = false;
    bool normalMappingEnabled = true;
    bool roughnessEnabled = true;
    bool aoEnabled = true;

    // Debug visuals
    bool showBoundingBoxes = false;
    bool showSDFDebug = false;

    // Debug: canonical view IDs in vulkan/includes/debug/DebugModes.hpp (0 = normal
    // render). Drives the raster solid/water shaders and the RT reference-path
    // forcing; both surfaces dispatch on the same IDs.
    int debugMode = 0;

    // Triplanar
    float triplanarThreshold = 0.12f;
    float triplanarExponent = 1.0f;

    // LoD rendering: per-chunk LoD ladder selection. Each chunk publishes
    // decimated levels (0 = full detail, N = coarsest). The GPU gate walks the
    // ladder hierarchically: a cell is subdivided while the camera is closer to
    // its AABB than `baseCell * level * lodBias`, and the first cell that is
    // not recursed into is the one drawn — exactly one rung per region (no
    // holes, no overlap). Larger values push coarser levels farther away (more
    // detail, more triangles) and smaller values switch to coarse meshes sooner
    // (fewer triangles). 0 = always coarsest, 64+ = effectively full detail
    // everywhere.
    float lodBias = 8.0f;

    // LoD rendering: maximum target LoD level the GPU band test may select for a
    // chunk. Clamps the coarsest level chosen, so geometry never renders coarser
    // than this level. 16 = effectively unlimited (chunk ladders rarely exceed
    // ~5 levels); lower values force only the finer chunk levels to be drawn
    // (more triangles, fewer coarse ancestors).
    int maxTargetLod = 16;

    // Tessellation
    bool tessellationEnabled = false; // user-enabled (was default false)
    bool shadowTessellationEnabled = false;
    // Deferred depth prepass for the solid pass (perf report 21 C2 gate).
    // true = depth prepass, then color with loaded depth (current behavior);
    // false = single forward pass, color writes depth. Consider false when
    // tessellation is on (vertex-bound regime: the prepass then doubles the
    // dominant stage). Forced on while solid RT paths run (the RT color
    // variants have no depth-write twin, and uncaptured depth would break
    // the water/composite passes that sample it).
    bool solidDepthPrepass = true;
    float tessellationFactor = 1.0f;
    float tessMaxDistance = 512.0f;
    float tessMinDistance = 1.0f;

    // Camera clip planes
    float nearPlane = 0.1f;
    float farPlane = 8092.0f;

    // Impostor rendering hand-off distance (m): beyond this distance the
    // vegetation (legacy billboards) and the SDF grass (Grass Raycast mode)
    // are drawn as pre-captured camera-facing impostor quads. 0 = disabled.
    // Shared by both modes so the far field hands off at one distance.
    float impostorDistance = 512.0f;

    // ── Hybrid RT (raster owns primary, CSM owns macro shadows, RT owns
    // secondary visibility: solid/water reflections, water refraction/
    // thickness, selective local/contact shadows) ──
    bool rtReflections = false;   // solid RT reflections (mirror/SSR rays)
    bool rtWaterReflections = false; // water RT reflections (own toggle so each ray path is switchable)
    bool rtRefractions = false;   // water refraction via Snell (IOR below); also carries RT thickness
    bool rtThickness = false;     // RT water thickness + Beer-Lambert absorption
    // Water-depth source for the shore-wave regions: true = ray-traced solid
    // bottom (inline ray query in the water TES, world-space drop); false =
    // raster only (solid scene depth + water volume back face, world-space
    // drops). Both modes produce a vertical world-space depth so the region
    // zones match. Requires RT to be enabled (falls back to raster otherwise).
    bool rtWaterDepth = false;
    bool rtLocalShadows = false; // selective RT contact shadows augmenting CSM (off = CSM-only, recommended)
    bool rtWaterPipeline = false; // water via async RT pipeline outputs (off = inline ray queries)
    int rtReflectionBounces = 1; // extra mirror rays when a reflection hits a reflective surface (0..3)
    float rtMaxReflectDist = 500.0f;  // reflection ray Tmax (world units)
    float rtMaxRefractDist = 300.0f;  // refraction ray Tmax (also deep-water thickness)
    float rtCoarseBoxSize = 48.0f;  // proxy boxes wider than this are "coarse": unreliable for refraction detail, treated as deep water/sky
    float rtMaxShadowDist = 12.0f;    // local shadow ray Tmax (contact range only)
    float rtRoughnessThreshold = 0.6f;// roughness above this skips RT reflections (env approx)
    // NOTE: water look (IOR, Beer-Lambert absorption, thickness cap) lives in
    // WaterSettings per water layer (Water Settings widget). The RT params UBO
    // still carries copies for the layer-unaware async pipeline path, synced
    // from water layer 0 in SceneRenderer::updateRTParams — there is only one
    // place to tweak them.
    // Proxies are thin tight slabs (vertex bounds ±0.15 m pad), so the ray
    // origin (biased along the shading normal) clears the fragment's own box
    // well below a meter; grazing rays then stay low enough to hit nearby
    // thin slabs instead of flying over them to sky. The old 2.0 m default
    // dated from full-cell-volume proxies and blinded flat-terrain mirrors.
    float rtSelfSkipDist = 0.05f;     // ignore proxy hits closer than this (own-box guard)
    // ── Ray-budget controls (runtime A/B, mirrored into RayTracingParams::rayParams) ──
    // rtRayScale: 0 = full-rate inline rays (reference), 1 = checkerboard
    //   half-rate (inline trace on even (x+y) pixels only, odd pixels keep the
    //   sky fallback — ~2x fewer ray queries). Applies to SOLID reflections;
    //   water reflections always trace full-rate (a skipped mirror is a
    //   missing mirror) and water refraction is covered by rtSingleRay.
    // rtRayContribMin: skip the inline ray when the lobe contribution is below
    //   this (terrain: blendedRefStrength*Fresnel*(1-rough); water: lobe mix).
    // rtSingleRay: water traces reflection XOR refraction stochastically
    //   (probability = Fresnel mix). Only the REFRACTION lobe honors the cut
    //   (it recovers from the raster bottom/sky); reflection always traces.
    //   off = dual-trace reference. debugModeForcesRtReference (see
    //   vulkan/includes/debug/DebugModes.hpp) forces reference (dual + full-rate)
    //   for traced-result debug views so diagnostics show full quality.
    int rtRayScale = 1;
    float rtRayContribMin = 0.02f;
    bool rtSingleRay = true;
    // ── Water-in-main migration (Phase 1): draw water chunks in the main
    // pass with the water-blend pipeline (see WaterRenderer.frag)
    // instead of the separate liquid pass. Default off (old path); consumed
    // by Phase-1b (draw routing + blend pipeline). No effect yet.
    bool waterInMainPass = false;

    // ── Water offscreen render scale (perf report 20 H9) ──────────────────
    // The water color/body/column pair, the water geometry depth and the
    // back-face depth render at this fraction of the swapchain size; the
    // composite upsamples (water color bilinear, water geometry depth with the
    // CLOSEST of the 2x2 taps so terrain in front is never punched through).
    // The water surface is a smooth translucent layer, so 0.5 is a cheap 4x cut
    // in shaded water pixels; 1.0 keeps full resolution. Applied when the water
    // targets are (re)created: immediately on change, or on a swapchain resize.
    float waterRenderScale = 1.0f;

    // ── Vegetation offscreen render scale (perf report 22 M12) ─────────────
    // The vegetation color + depth targets render at this fraction of the
    // swapchain size; the composite upsamples (color bilinear; the depth takes
    // the CLOSEST of the 2x2 taps when scaled, so vegetation silhouettes are
    // never eroded by the averaged sky value). Vegetation detail is leaf-scale,
    // which the impostor path already abandons at range, so 0.5 on the dense
    // presets is a 4x cut in shaded vegetation pixels. Applied when the
    // targets are (re)created: immediately on change, or on swapchain resize.
    float vegetationRenderScale = 1.0f;

    // ── Main material texture-array resolution (perf report 23 C1) ─────────
    // The five material arrays (albedo/normal/bump/roughness/ao) are RGBA8
    // with a full mip chain at Size x Size per layer and are the largest
    // single image commitment in the engine (~750 MB at 1024 with the
    // content-derived layer count). 1024 is the default; the low presets set
    // 512, which quarters the commitment. Applied on the frame the value
    // changes: one device idle, then the layers are re-uploaded and the
    // mixer's generated layers refreshed (the array semantics — layer
    // indices, materials, mixer targets — are resolution-independent).
    int textureArraySize = 1024;
};
