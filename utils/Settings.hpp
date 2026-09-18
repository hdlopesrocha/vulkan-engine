#pragma once

class Settings {
public:
    void resetToDefaults() {
        *this = Settings{};
    }

    // Global toggles
    bool enableShadows = true;
    // Toggle rendering of the main solid scene (terrain/meshes)
    bool renderSolid = true;
    bool waterEnabled = true;
    bool vegetationEnabled = true;
    bool wireframeMode = false;
    bool waterWireframeMode = false;
    bool normalMappingEnabled = true;
    bool roughnessEnabled = true;
    bool aoEnabled = true;

    // Debug visuals
    bool showDebugCubes = false;
    bool showBoundingBoxes = false;
    bool showSDFDebug = false;

    // Input settings
    bool flipKeyboardRotation = false;
    bool flipGamepadRotation = false;
    float moveSpeed = 2.5f;
    float angularSpeedDeg = 45.0f;

    // Debug
    int debugMode = 0;

    // Triplanar
    float triplanarThreshold = 0.12f;
    float triplanarExponent = 1.0f;

    // LoD rendering: per-chunk LoD ladder selection. Each chunk publishes
    // decimated levels (0 = full detail, N = coarsest). The GPU band test keeps
    // entry level k for dist in [k, k+1) * chunkBase * lodBias, so larger
    // values push coarser levels farther away (more detail, more triangles)
    // and smaller values switch to coarse meshes sooner (fewer triangles).
    // 0 = always coarsest, 64+ = effectively full detail everywhere.
    float lodBias = 8.0f;

    // LoD rendering: maximum target LoD level the GPU band test may select for a
    // chunk. Clamps the coarsest level chosen, so geometry never renders coarser
    // than this level. 16 = effectively unlimited (chunk ladders rarely exceed
    // ~5 levels); lower values force only the finer chunk levels to be drawn
    // (more triangles, fewer coarse ancestors).
    int maxTargetLod = 16;

    // Tessellation
    bool tessellationEnabled = true; // user-enabled (was default false)
    bool shadowTessellationEnabled = true;
    bool adaptiveTessellation = true;
    float tessellationFactor = 1.0f;
    float tessMaxDistance = 512.0f;
    float tessMinDistance = 1.0f;

    // Present mode
    bool vsyncEnabled = true;

    // Camera clip planes
    float nearPlane = 0.1f;
    float farPlane = 8092.0f;

    // Impostor rendering: vegetation beyond this distance is drawn as a pre-captured
    // camera-facing quad.  Set to 0 to disable (default: disabled).
    float impostorDistance = 512.0f;

    // ── Hybrid RT (raster owns primary, CSM owns macro shadows, RT owns
    // secondary visibility: solid/water reflections, water refraction/
    // thickness, selective local/contact shadows) ──
    bool rtReflections = true;   // solid + water RT reflections (sky on miss/off)
    bool rtRefractions = true;   // water refraction via Snell (IOR below)
    bool rtThickness = true;     // RT water thickness + Beer-Lambert absorption
    bool rtLocalShadows = false; // selective RT contact shadows augmenting CSM (off = CSM-only, recommended)
    bool rtWaterPipeline = false; // water via async RT pipeline outputs (off = inline ray queries)
    float rtMaxReflectDist = 500.0f;  // reflection ray Tmax (world units)
    float rtMaxRefractDist = 300.0f;  // refraction ray Tmax (also deep-water thickness)
    float rtCoarseBoxSize = 48.0f;  // proxy boxes wider than this are "coarse": unreliable for refraction detail, treated as deep water/sky
    float rtMaxShadowDist = 12.0f;    // local shadow ray Tmax (contact range only)
    float rtRoughnessThreshold = 0.6f;// roughness above this skips RT reflections (env approx)
    // NOTE: water look (IOR, Beer-Lambert absorption, thickness cap) lives in
    // WaterParams per water layer (Water Settings widget). The RT params UBO
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
    //   half-rate (inline trace on even (x+y) pixels only, odd pixels reuse the
    //   pipeline/sky fallback — ~2x fewer ray queries).
    // rtRayContribMin: skip the inline ray when the lobe contribution is below
    //   this (terrain: blendedRefStrength*Fresnel*(1-rough); water: lobe mix).
    // rtSingleRay: water traces reflection XOR refraction stochastically
    //   (probability = Fresnel mix) instead of always both (~2x fewer rays);
    //   off = dual-trace reference. Any rtDebugView except 59 (the budget
    //   mask itself) forces reference (dual + full-rate) so diagnostics show
    //   full quality.
    int rtRayScale = 1;
    float rtRayContribMin = 0.02f;
    bool rtSingleRay = true;
    // ── Water-in-main migration (Phase 1): draw water chunks in the main
    // pass with the WATER_MODE=1 water-blend pipeline (see main.frag)
    // instead of the separate liquid pass. Default off (old path); consumed
    // by Phase-1b (draw routing + blend pipeline). No effect yet.
    bool waterInMainPass = false;
    // RT debug views (0=off; also drives ubo.debugParams extensions in shaders):
    //  50=RT reflection only, 51=RT refraction only, 52=thickness,
    //  53=Fresnel, 54=absorption, 55=CSM-only, 56=RT-local-only, 57=CSM+RT combined
    int rtDebugView = 0;
};
