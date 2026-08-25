#pragma once

class Settings {
public:
    void resetToDefaults() {
        *this = Settings{};
    }

    // Global toggles
    bool enableShadows = false;
    // Toggle rendering of the main solid scene (terrain/meshes)
    bool renderSolid = true;
    bool waterEnabled = false;
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
    bool tessellationEnabled = false;
    bool shadowTessellationEnabled = false;
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
};
