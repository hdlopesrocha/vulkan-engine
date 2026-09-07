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
    bool rtWaterPipeline = true; // water via async RT pipeline outputs (off = inline ray queries)
    float rtMaxReflectDist = 500.0f;  // reflection ray Tmax (world units)
    float rtMaxRefractDist = 300.0f;  // refraction ray Tmax (also deep-water thickness)
    float rtMaxWaterThickness = 6.0f; // clamp for RT hit thickness (kills far-hit blackouts)
    float rtMaxShadowDist = 12.0f;    // local shadow ray Tmax (contact range only)
    float rtRoughnessThreshold = 0.6f;// roughness above this skips RT reflections (env approx)
    float rtWaterIOR = 1.333f;        // physical water IOR
    float rtAbsorption[3] = {0.35f, 0.12f, 0.08f}; // Beer-Lambert RGB coefficients
    float rtAbsorptionScale = 1.0f;   // thickness multiplier for absorption viz/tuning
    float rtSelfSkipDist = 2.0f;      // ignore proxy hits closer than this (own-box guard)
    // RT debug views (0=off; also drives ubo.debugParams extensions in shaders):
    //  50=RT reflection only, 51=RT refraction only, 52=thickness,
    //  53=Fresnel, 54=absorption, 55=CSM-only, 56=RT-local-only, 57=CSM+RT combined
    int rtDebugView = 0;
};
