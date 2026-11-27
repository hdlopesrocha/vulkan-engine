#pragma once

#include "WindParticlesSettings.hpp"
#include "WindRaymarchSettings.hpp"

class Settings {
public:
    // Grass representation selector (Settings widget: "Grass"):
    //   Bladed     = triangle blade tufts + captured impostors (only path)
    //   None       = no grass
    // The legacy billboard quads and SDF raymarched grass are retired.
    enum class GrassMode { Bladed = 0, None = 1 };

    void resetToDefaults() {
        *this = Settings{};
    }

    // Global toggles
    bool enableShadows = true;
    // SDF shadow raymarch (octree path). The ray always inherits the primary
    // hit's effective LOD; these controls only bound cost and penumbra.
    // softness: 0 = hard, otherwise k in visibility = min(v, k * distance / t).
    // maxDistance: 0 = use the camera far plane (bounded by the shader cap).
    // maxSteps: early-out step budget (1..1024, shader cap 256).
    float shadowSoftness = 8.0f;
    float shadowMaxDistance = 2048.0f;
    int shadowMaxSteps = 256;
    // Volumetric clouds (sky raymarch + ground shadows + reflections).
    // Minimal preset disables; Maximum enables. Default off: the sky
    // gradient renders without the volumetric march.
    bool cloudsEnabled = false;
    // Toggle rendering of the main solid scene (terrain/meshes)
    bool renderSolid = true;
    // Bladed grass (triangle tufts + impostors). Derived from grassMode every
    // frame (MyApp): true unless GrassMode::None.
    bool vegetationEnabled = false;
    // ── Wind-trail particles (WindRenderer) ────────────────────────────────
    // GPU-advected particles that follow the shared wind field and draw fading
    // world-space trails. On by default (it is the wind-field visualization);
    // toggle it off here or in the Wind Trails widget. Trail duration/color/
    // width and the particle count live in `windParticles`
    // (WindParticlesWidget); the wind itself is the shared Wind widget field.
    bool windParticlesEnabled = true;
    WindParticlesSettings windParticles;
    // WindRaymarchRenderer: renders |wind| == targetSpeed as a raymarched
    // isosurface shape. Off by default (a diagnostic/visualization mode).
    bool windRaymarchEnabled = false;
    WindRaymarchSettings windRaymarch;
    // Active grass representation (see GrassMode). Default: bladed grass.
    GrassMode grassMode = GrassMode::Bladed;
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
    // smoke). The sky cloud raymarch has its own Volumetric Clouds toggle.
    // Minimal preset turns this off.
    bool rayMarchingEnabled = false;
    bool normalMappingEnabled = true;
    bool roughnessEnabled = true;
    bool aoEnabled = true;

    // ── Water droplets (GPU particle system + SDF ray-marched rendering) ──
    // 2048 particles are simulated on the GPU (spatial-hash broad phase,
    // sphere/SDF contacts, impulse solver) and rendered by ray marching their
    // SDF spheres with refraction + reflection. The simulation is re-seeded
    // inside `dropletSpawnRadius` every `dropletResetSeconds`, centred at
    // (0, terrainHeight + dropletSpawnHeight, 0) — i.e. a ball of droplets
    // hovering above the height map that then falls onto and into it, colliding
    // with the terrain SDF and with each other.
    // Physics and rendering are independent: the values below never change the
    // terrain's rendering LOD and the render LOD never changes the collision.
    bool dropletsEnabled = true;       // master gate (sim + draw)
    int dropletCount = 2048;           // simulated particles (<= capacity 8192)
    float dropletSpawnRadius = 128.0f; // spawn volume radius (m)
    float dropletSpawnHeight = 600.0f; // spawn centre height ABOVE the terrain (m)
    float dropletRadius = 2.5f;        // particle sphere radius (m), sim + render
    float dropletResetSeconds = 30.0f; // full re-seed interval (s)
    float dropletGravity = 9.81f;      // m/s^2 (downward)
    float dropletRestitution = 0.25f;  // bounce factor at SDF/water contacts
    float dropletFriction = 0.35f;     // tangential (Coulomb-clamped) friction
    float dropletDamping = 0.02f;      // per-second linear drag
    float dropletIor = 1.333f;         // refraction index for the render
    int dropletSolverIterations = 3;   // pairwise impulse iterations per substep
    int dropletSubsteps = 4;           // fixed substeps per frame (dt = frame/n)
    bool dropletStats = false;         // read contact/overflow counters (log)

    // ── Scene server (serve THIS application's live scene) ─────────────────
    // When enabled, MyApp starts a SceneServer bound to its own LocalScene:
    // remote clients (the bundled web viewer, or another engine instance in
    // --remote mode) receive the chunk ladder and every live octree edit, the
    // same protocol bin/server speaks. Off by default; the port must be free.
    bool serveScene = false;
    int serveScenePort = 8090;

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
    // Terrain is rendered exclusively by the full-screen octree SDF raymarch
    // (Solid and Water layers in one pass); there is no terrain rasterizer and
    // no terrain chunk VBO/IBO path. The raymarch applies the same tessellation
    // height displacement as the retired raster path by folding it into the
    // SDF (Settings::tessellationEnabled).
    //
    // Screen-space hit tolerance in pixels; 0 = off; 0.25 is the recommended
    // start; larger values cut distant/grazing march steps at a sub-pixel
    // accuracy cost.
    float octreeRaymarchScreenEps = 0.0f;
    // True raymarched displacement master switches (material fields live in
    // MaterialProperties; these gate the new path globally).
    bool displacementEnabled = true;
    bool displacementAOEnabled = true;
    float tessellationFactor = 1.0f;
    float tessMaxDistance = 512.0f;
    float tessMinDistance = 1.0f;

    // Camera clip planes
    float nearPlane = 0.1f;
    float farPlane = 8092.0f;

    // Impostor rendering hand-off distance (m): beyond this distance bladed
    // grass tufts are drawn as pre-captured camera-facing impostor quads.
    // 0 = disabled (every tuft stays geometric).
    float impostorDistance = 512.0f;

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
