# Hardware Ray-Tracing Renderer — Architecture

The engine's primary solid/water visibility path is a hardware-accelerated
Vulkan ray tracer (`VK_KHR_acceleration_structure` +
`VK_KHR_ray_tracing_pipeline`). The legacy rasterizer stays available behind
the `RT_RENDERER_ENABLED` runtime switch for comparison.

```
Camera -> Ray Generation -> TLAS traversal -> Closest/Any Hit
  -> Material evaluation (existing SSBOs/textures) -> Lighting / reflection /
     refraction (secondary TLAS traversals) -> HDR scene image (+ NDC depth)
     -> existing post-process / presentation
```

## New components

| File | Responsibility |
|---|---|
| `vulkan/renderer/RayTracingSupport.hpp` | RT constants (`MATERIAL_WATER` bit, ray types, IOR, recursion limits), physical-device feature detection (`queryDeviceSupport`), KHR entry-point loader (`Functions`), buffer-address helper |
| `vulkan/renderer/AccelerationStructureManager.{hpp,cpp}` | Incremental BLAS/TLAS system (below) |
| `vulkan/renderer/RayTracingRenderer.{hpp,cpp}` | HDR output images, RT descriptor set, RT pipeline + SBT, per-frame dispatch, stats, ImGui panel, debug views |
| `shaders/rt_basic.rgen` | Camera rays, primary trace, HDR + NDC-depth write, debug visualization |
| `shaders/rt_radiance.rmiss` | Sky/atmosphere environment for primary + secondary misses |
| `shaders/rt_shadow.rmiss` | Shadow-ray miss (lit) |
| `shaders/rt_solid.rchit` | Solid material eval, RT shadows, Fresnel reflection |
| `shaders/rt_water.rchit` | Water: Fresnel reflection, Snell refraction, ray-derived thickness, Beer–Lambert |
| `shaders/rt_alpha.rahit` | Alpha-tested vegetation (`ignoreIntersectionEXT` below threshold) |
| `shaders/includes/rt_common.glsl` | RT descriptor layout, payloads, vertex fetch, sky, Schlick, Beer–Lambert |
| `shaders/includes/rt_shading.glsl` | Material blend, direct light + shadow rays, albedo/roughness fetch |

## Device setup (`VulkanApp::createLogicalDevice`)

- Probes `queryDeviceSupport()` (extension list + `Features2` chain) — never
  assumes RT from the Vulkan version.
- When usable, enables `VK_KHR_acceleration_structure`,
  `VK_KHR_ray_tracing_pipeline`, `VK_KHR_deferred_host_operations`
  (+ `VK_KHR_ray_query` when supported), `bufferDeviceAddress`,
  `scalarBlockLayout`, and chains
  `VkPhysicalDeviceRayTracingPipelineFeaturesKHR` /
  `VkPhysicalDeviceAccelerationStructureFeaturesKHR` (+ ray-query) into
  `VkDeviceCreateInfo::pNext`, composed with the existing descriptor-buffer
  chain.
- Loads all KHR entry points into `VulkanApp::rtFunctions`; renderers branch
  on `supportsRayTracing()`. Without RT the engine runs exactly as before.

## Acceleration structures (no full rebuilds, no readbacks)

- **BLAS**: one per active `IndirectRenderer` slot (chunk), for the solid and
  water layers. Geometry points **into** the shared packed vertex/index pools
  via device addresses + (`baseVertex`, `firstIndex`) sub-ranges — no geometry
  copies. Requires the pools to carry `SHADER_DEVICE_ADDRESS |
  ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | STORAGE_BUFFER` usage
  (added in `IndirectRenderer::initSlots`; harmless on non-RT devices).
- **TLAS**: one instance per BLAS, identity transform (scene is world-space).
  `instanceCustomIndexEXT` packs layer routing: bit 31 = `MATERIAL_WATER`,
  bit 30 = vegetation/alpha-test, bits 0–29 = slot id. Solid instances use
  mask `0x01` / SBT record 0, water mask `0x02` / record 1, so primary,
  shadow (`traceMask`), reflection and transmission rays can include/exclude
  the water layer by mask.
- **Dirty tracking**: `syncFromScene()` snapshots active slot geometries
  (`collectActiveSlotGeometries()`) and diffs them on the CPU. Only
  added/modified slots rebuild their BLAS; the TLAS rebuilds only when
  membership or a BLAS address changed. Steady state = zero AS builds.
- **LoD rung selection**: BLASes exist for every resident rung, but the TLAS
  instances only the rung `indirect.comp` would draw per region. The CPU
  mirror (`lodRungSelected`, fuzz-verified bit-faithful over 200k cases
  against the shader formula) evaluates the same clipmap band gate from the
  same per-slot inputs (rung, cube length, column anchor, ladder depth) and
  the same camera/bias caps. Without it, coarse ancestor rungs smear over
  fine detail. A changed selection (camera crossing a rung boundary) rebuilds
  the TLAS; a static camera rebuilds nothing. Frustum culling is
  intentionally NOT mirrored — off-screen geometry must stay traversable for
  reflections.
- **Freshest-TLAS dispatch**: TLAS objects are per frame slot, but every
  frame's descriptor set points at the newest built object across all slots
  (`tlasFresh()`). Rebuilding only the current slot while other slots serve
  stale rung selections makes frames permanently alternate between old and
  new detail (chunk flicker) after every selection change — all sets are
  re-pointed together on dirty frames instead. The overlap-column counter in
  the overlay (`overlap`) measures multi-rung columns in the live set and
  must read 0.
- **Synchronization** (all Synchronization2): transfer/compute writes →
  `AS_BUILD_READ` (pools) → BLAS builds → barrier → TLAS build →
  `RAY_TRACING_SHADER_READ` barrier → `vkCmdTraceRaysKHR`, all in one command
  buffer. Across submits there is deliberately *nothing shared and mutable*:
  scratch, TLAS object/storage, TLAS instances and slot meta are all
  per-frame-slot (reused only after that slot's frame fence, exactly like the
  engine's per-slot cull buffers), and BLAS storage is freshly allocated per
  rebuild with deferred retirement — so no cross-submit chaining primitive
  is needed at all. (An earlier revision used a persistent `VkEvent` for
  this; the layer's event state tracking proved too brittle across frames,
  so the design was changed to eliminate sharing instead.) No
  `vkDeviceWaitIdle` / `vkQueueWaitIdle` in the loop.

## Materials (reused, not duplicated)

Hit shaders fetch the hit triangle from the packed pools (uint view over the
64-byte `Vertex`: pos[0..2] color[3..5] uv[6..7] normal[8..10]
brushIndex[11] hsv[13..15]) using the per-layer slot-meta SSBO
(`baseVertex`/`firstIndex`/counts indexed by slot id), interpolate
position/normal/UV/HSV by barycentrics, and evaluate the **existing**
`Materials` SSBO + albedo/normal/height/roughness/AO texture arrays with the
same per-corner barycentric blend as the rasterizer. Water params come from
the existing water-params SSBO indexed by brush (same rule as `water.frag`).

Raster equivalence (`rt_material.glsl`, same formulas as `main.frag`):
triplanar weights from the geometric normal with the scene threshold/
exponent, per-corner triplanar UVs with per-material scales and sign rules,
height sampling with invert/width flips, TES displacement `(h-0.5)*scale`
applied to the hit point, triplanar or UV albedo/roughness/AO, normal maps
with per-material conventions, exact ambient/diffuse/specular/AO/HSV math,
and the exact env mix factor (traced scene in place of the cubemap).
Known deltas: no derivatives exist in ray tracing, so the non-triplanar
normal-map TBN uses a stable up-cross basis instead of the rasterizer's
dFdx basis; traversal stays on the base mesh (silhouette micro-detail from
tessellation displacement differs, shading point is displaced equivalently);
brush PAINT/REMOVE overrides are raster composite features and don't alter
RT material lookup. Displacement honors the global tessellation toggle, and
all secondary origins lift clear of the base mesh on both sides of a
negative (inward) displacement, so disabled tessellation never leaves
bump-shaped shading or bump-following self-shadowing behind.

## Water (first-class optical material)

- Identified by the `MATERIAL_WATER` instance bit → water hit group
  (`rt_water.rchit`), never treated as opaque.
- Reflection: `R = reflect(I, N)`, Schlick Fresnel (`F0 ≈ 0.02`, IOR 1.333),
  traced above the surface through the real TLAS (off-screen geometry
  reflects correctly).
- Refraction: Snell's law via `refract()`, IOR 1.333, TIR → full reflection.
  The transmission ray (solids-only mask so the entry surface never
  self-hits) finds the terrain behind the water; its `hitT` **is** the
  ray-derived `waterThickness = exit − entry`.
- `T = exp(-σ·thickness)` (Beer–Lambert, configurable RGB σ) attenuates the
  refracted terrain; shallow/deep colors mix by `1 − exp(−thickness/τ)`.
  Shallow water (small thickness) and deep water attenuate visibly
  differently; submerged terrain is shaded normally then attenuated.
- Bounded recursion: PRIMARY → REFLECTION(1) → REFRACTION(1), pipeline
  recursion ≤ 4, `maxDepth` configurable 1–4 in the UI.

## Payload model (bounded recursion + glslang codegen rule)

- Two payloads, both small: radiance (color, hitT, normal, fresnel,
  material/slot/primitive ids, water thickness, bounce count,
  reflect/refract weights, shadow, world pos) and shadow (occlusion float).
- PRIMARY → REFLECTION(1) → REFRACTION(1); `maxDepth` configurable 1–4 in the
  UI; pipeline recursion cap is 4.
- **Critical glslang rule (verified in SPIR-V disassembly): `traceRayEXT`'s
  payload argument must be a module-scope `rayPayloadEXT` variable.**
  Function-local payloads are silently miscompiled to the incoming payload
  (no compile error, no validation error — nested results read back as
  zeros). Nested rays therefore use payload FORWARDING: each closest-hit
  saves its inputs in locals, resets `payload.bounceCount` to its own depth,
  traces into the global `payload`, reads the nested result immediately, and
  writes back its final fields (restoring depth + primary identity) before
  returning. Shadow rays use a dedicated global payload. When the bounce
  budget is exhausted, reflection/refraction fall back to the sky environment
  (never black).

## Shadows, sky, vegetation

- Direct lighting uses ray-traced shadow rays
  (`TERMINATE_ON_FIRST_HIT | OPAQUE | SKIP_CLOSEST_HIT`, directional light,
  Blinn-Phong specular consistent with `main.frag`).
- Miss returns the sky gradient + sun disk/flare from the existing
  `SkyUniform` (never black) for primary/reflection/refraction rays.
- Vegetation: any-hit alpha test (`rt_alpha.rahit`) samples the albedo array
  and `ignoreIntersectionEXT`s below 0.5. The TLAS currently holds
  solid+water instances; raster vegetation keeps compositing over the RT image
  against the RT NDC depth while the vegetation-instance feed is completed.

## Output + integration

- Per-frame HDR color (`R16G16B16A16_SFLOAT`, alpha = hit coverage so the
  sky composite keeps working) + NDC-depth companion (`R32_SFLOAT`, same
  space as the raster depth buffer so post-process obstacle tests compare
  correctly). `GENERAL` during tracing → `SHADER_READ_ONLY_OPTIMAL` for the
  existing `PostProcessRenderer`, which is otherwise untouched.
- `MyApp::preRenderPass` records AS sync/build + dispatch on the main command
  buffer before the async tasks; when RT is primary the legacy solid depth/
  color, sky-fullscreen, wireframe, back-face and water-geometry **draws** are
  skipped (clears + all timeline signals kept, so the frame graph cannot
  deadlock). Shadow maps, culling, sky equirect, vegetation and debug passes
  keep running for the overlays that still consume them.
- Modes: `Legacy` (raster only) / `Ray tracing` (default) / `RT debug`.
  Debug views: distance, world normal, material ID, instance/primitive,
  Fresnel, water thickness, reflection/refraction weight, shadow, bounces.

## Performance

- No per-frame TLAS rebuilds; BLAS builds only for dirty slots; one shared
  scratch buffer; SBT/descriptors created once (TLAS binding rewritten only
  when the TLAS handle changes; frame constants stream via UBO memcpy).
- Small payloads (radiance + shadow); secondary rays gated by Fresnel/material
  thresholds and `maxDepth`; early termination on first shadow hit.
- Stats in the overlay + Ray Tracing panel: BLAS/TLAS counts, per-frame/total
  builds, AS CPU ms, dispatch record ms, primary rays, output size.

## Correctness tests (scenes/debug)
Use `RT debug` mode + the views below (all runtime, no rebuild):

- **A solid reflection**: metallic object + off-screen occluder → Reflection
  view shows the occluder; shaded view shows it mirrored.
- **B water reflection**: terrain beside water → water shows terrain.
- **C shallow water**: terrain just under water → Water-thickness view is dark
  (small values), weak attenuation.
- **D deep water**: terrain far below → bright thickness values, strong
  Beer–Lambert darkening toward `deepColor`.
- **E submerged object**: object under water → refracted + attenuated, depth
  tint matches surrounding thickness.
- **F behind-camera reflection**: object behind the camera near water/metal →
  visible in the reflection (impossible for SSR).
- **G moving water**: animate water time → reflections update every frame
  (no history/ghosting; the path is history-free by construction).
- **H vegetation**: foliage near water → alpha-tested visibility; raster
  composite path shows it over RT until TLAS vegetation instances land.

## Validation status (headless e2e)

`rt_e2e_test` (kept out of tree; rebuild under `/tmp` if needed) exercises the
real backend without a window: device chain with the exact RT extensions +
features, 2-BLAS/2-instance TLAS builds, the real 6-SPIR-V pipeline + SBT +
descriptors, dispatch, and readback — all under validation layers:

- Device creation with the RT chain: PASS (RADV REMBRANDT, Vulkan 1.4).
- AS builds + pipeline + SBT + dispatch: 0 validation errors, instant.
- Sky miss (radiance): gradient + sun, alpha 0, depth 1.0.
- Solid hit: lit material color, alpha 1, correct NDC depth.
- Water entering hit: Beer–Lambert deep attenuation + Fresnel sky reflection.
- Water grazing/exiting hit: Fresnel reflection, no black pixels.
- All RT SPIR-V passes `spirv-val --target-env vulkan1.3`.

Known interim limitations: the raster brush-liquid overlay is skipped in RT
mode (its set-2 inputs are only created by the skipped raster water pass;
an empty timeline-shaped submit preserves the `tlBrushLiquid` signal so the
composite cannot deadlock) — brush solid/color previews are unaffected;
vegetation instances are not yet in the TLAS
(raster vegetation composites over the RT image against the RT NDC depth;
the any-hit alpha path is implemented and validated at the shader level);
no temporal accumulation (the path is history-free by construction, so no
ghosting — accumulation is future work); the solid360 cubemap and shadow-map
passes keep running for the overlays that still consume them.
