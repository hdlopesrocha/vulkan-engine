# True raymarched displacement mapping

Status: implemented in the octree SDF raymarcher
(`shaders/renderer/octree/OctreeRaymarch.frag`), its CSM terrain caster
(`shaders/renderer/octree/OctreeRaymarchShadow.frag`) and the shared field
definition `shaders/includes/octree/DisplacementField.glsl`.

This is real geometric displacement: the ray/surface intersection is moved, so
silhouettes, depth, shadows, normals, AO and secondary rays all see the same
displaced zero set. It is not normal mapping, bump mapping or parallax
occlusion mapping, and it does not require tessellated meshes or permanent
octree subdivision.

## 1. Mathematical formulation

The octree stores one signed value per cell corner. The field used by the
renderer is the trilinear reconstruction of those eight corners:

```
d_base(p) = trilinear(corner SDFs of the containing cell, p)
```

A material height sample produces a signed world-space offset `h(p)`:

```
h_legacy(p) = (height(p) - 0.5) * tessHeightScale
h_true(p)   = clamp((height(p) - 0.5 + offset) * amplitude * strength,
                    -bounds, +bounds)
```

The displaced surface is the zero level set of

```
F(p) = d_base(p) - h(p)
```

The sign convention follows the old raster tessellation path: positive `h`
moves the surface along the base outward normal, negative recesses it. The
vertex displacement convention (`p + n * h`) and the field convention
(`d_base - h`) are therefore identical to first order. The specification's
`d_original + displacement` is the same equation with their displacement
defined as `-h`.

The legacy mode is still available and is bit-compatible with the previous
tessellation bump: `mappingEnabled` plus the global tessellation toggle uses
`(height - 0.5) * tessHeightScale` with a `0.5 * |scale|` bound. The new path
is enabled independently per material by a non-zero
`displacementStrength`/`displacementAmplitude` and the global
`Settings::displacementEnabled` flag.

### Material controls

`MaterialProperties` (CPU) and the packed `Material` SSBO (GPU) carry:

| Control | Field | Meaning |
|---|---|---|
| Enable | `displacementStrength > 0` | 0 disables; 1 applies the full amplitude |
| Amplitude | `displacementAmplitude` | Signed world units at the height extremes (negative inverts peaks/pits) |
| Frequency | `displacementFrequency` | UV multiplier on top of the per-vertex texture scale; also the declared feature wavelength for the march |
| Strength | `displacementStrength` | Blend between undisplaced (0) and full amplitude (1) |
| Offset | `displacementOffset` | Normalized height bias (+0.5 raises the neutral surface by half the amplitude) |
| Bounds | `displacementBounds` | Optional clamp on `|h|` in world units (0 = `|amplitude| * strength`) |
| Texture | `displacementTexture` | Height layer used for displacement (-1 = the material's own layer, otherwise resolved through the brush map) |
| Pattern | `displacementPattern` | 0 = texture, 1 = triangles, 2 = hexagons, 3 = Perlin, 4 = Voronoi, 5..9 = platonic solids |
| Pattern scale | `displacementPatternDisplacement` | Peak world-unit scale displacement |
| Pattern repeat | `displacementPatternRepeat` | World-unit size of one scale cell |
| Segment | `displacementSegmentTime/Velocity/Intensity` | Last dragged brush segment; orientation, animation phase and intensity |

The packed `Material` carries `displacementParams`, `patternParams` and
`segmentParams` and is 176 bytes. Two previously reserved floats carry the
texture bounds clamp and texture index. A material with
`displacementStrength == 0` (the default) behaves exactly as before.

## 2. Conservative marching

A displaced field is not a signed distance field. The march therefore uses a
two-region strategy.

### Outside the displacement band

When `|d_base| > maxAbs(h)`, the sign of `F` is the sign of `d_base` because
`|h| <= maxAbs < |d_base|`. The guaranteed-positive remainder

```
safe = |d_base| - maxAbs
```

can be advanced using the base field's own Lipschitz bound `L_base`:

```
step = safe / L_base
```

The engine already caches `1/L_base` per node (`OctreeDerive.comp` writes
`hsv1.z` from the corner differences). Cells without the payload use the
classical 0.5 factor. This region cannot skip a crossing.

### Inside the displacement band

Inside the band the field is not a distance field. The implementation uses

```
L_F <= L_base + L_h
L_h ~= 2*pi*maxAbs / wavelength
step = |F| / L_F
```

`wavelength` is the material's declared world-space feature size:

* new path: `1 / max(|uvScale| * frequency)`,
* legacy path: `nodeSize / (tessLevel * 16)` (the raster vertex spacing),

floored by sixteen pixels at the sample distance. The step is clamped to
`[feature/8, feature/2]`, so the march samples at least twice per declared
feature and never degenerates into micro-steps near the crossing. The sixteen
pixel floor keeps an in-band step at 2–8 pixels instead of dozens of sub-pixel
samples; the latter was fast enough to trip the GPU watchdog on the integrated
target. Displaced primary rays also use a 512-step hard budget (instead of the
base 1024); unresolved rays discard rather than risking a device-lost.
Every step is additionally clamped to the current child-cell exit, so entering
a stored Surface child is never skipped.

The declared frequency is the band limit. Displacement detail finer than that
(and finer than a pixel) is treated as filtered detail, in the same sense a
texture mip chain filters sub-pixel texels. The report calls this out under
limitations: if a height map contains a one-texel spike far narrower than the
declared wavelength, the march can miss it, exactly as a rasterizer would miss
sub-pixel geometry. The material frequency control is the contract that lets a
scene author keep the displacement band-limited.

### Intersection refinement

When a sample pair brackets a crossing with `F(t) > 0 >= F(t + step)`, the
march establishes `[lo, hi]` and runs a safeguarded root finder:

1. the first two iterations bisect (guaranteed bracket reduction);
2. later iterations take a secant step clamped to the bracket;
3. a secant step outside the bracket falls back to bisection;
4. the loop exits on `|F| <= max(eps * 0.05, 1e-6)` or
   `hi - lo <= max(eps * 0.05, 1e-6)`;
5. the returned position is the bracket midpoint; the residual is evaluated
   again for the debug view.

A hard 16-iteration cap is a safety limit, not the precision criterion. The
same node is used for the bracket, so the field expression is identical at
every refinement sample.

### Virtual subdivision

No octree node is allocated for displacement. The traversal still walks the
stored Surface hierarchy, and the stored cell AABB gives the conservative
spatial interval. The displacement band does two things:

* Empty/Solid child skipping is suspended for displaced materials, because the
  offset can push the surface across a cell border;
* the in-band step is clamped to the material feature wavelength, which can be
  much finer than the cell, so detail finer than the octree resolution is
  resolved analytically inside the cell ("virtual" refinement).

The distance-adaptive octree LOD gate is unchanged; details that are below the
screen-space target are already sub-pixel and do not need extra node descent.

## 3. Triplanar displacement

Displacement uses the same mapping modes as the material:

* **Triplanar**: world-space XY/XZ/YZ projections with the existing
  dead-zone/exponent weights from the geometric normal. The three height
  samples are blended *before* the displacement is formed, so projection
  transitions remain continuous and no UV seam is introduced.
* **Single projection / Angular / Custom**: the dominant mapped UV is sampled,
  matching the old displacement path.

The per-vertex UV scale (`OctreeNode::data`, painted by the
`TextureCoordPainter`) is used, multiplied by the material frequency. Negative
world coordinates are handled by the existing mirrored projection convention.
Changing the vertex scale changes the world-space feature wavelength used by
the march, so the sampling density follows the texture density. The field only
depends on world position + material, not on the camera, so camera and shadow
rays see the same surface.

`displacementTexture` selects another material's height layer through the brush
map; the owner material's invert-U/V flags are still applied. The special value
`-2` selects a procedural world-space 2D Perlin field in the same projection
frame (the "procedural displacement function" path), so no height texture is
required.

### 3.1 Secondary SDF scale patterns

The alternative source is a **GPU-only secondary SDF**: a procedural scale
pattern that is added to the displacement field but never stored in the
octree. A material selects it with `displacementPattern`:

* **Triangular scales** — a staggered grid of isosceles scales. Each scale's
  tip points along `-velocity`; its base is on the trailing side.
* **Hexagonal scales** — a staggered grid of pointed hexagons using the same
  `-velocity` orientation.
* **Perlin noise** — animated 4-octave 3D FBM, advected by the full segment
  velocity vector.
* **Voronoi cells** — animated 2D cellular F1 field in the same cylindrical
  frame, advected by velocity and time.
* **Platonic solids** — tetrahedron, cube, octahedron, dodecahedron and
  icosahedron reliefs, one per pattern cell. Each cell spins around the
  velocity axis with a cell-specific phase while the lattice itself is
  advected by velocity and time.

The pattern is evaluated in a cylindrical frame around the brush segment
velocity:
`u = dot(p, axis)`, `v = angle(p - axis * dot(p, axis)) * radius`. Using a
stable world-frame cylinder instead of a per-sample tangent frame is what
keeps the scalar field continuous across octree cells. `u` animates from
`appTime - segmentTime` and `v` uses the local radius, so triangles/hexagons
stay cell-sized around the trail. The pattern height is multiplied by
`displacementPatternDisplacement`, `displacementStrength` and the segment
intensity, then clamped to the same conservative bound used by the march.

`BrushEntry` carries the authoring fields and the last segment
`(secondarySdfSegmentTime, secondarySdfVelocity, secondarySdfIntensity)`.
Keyboard and gamepad brush translation compute velocity as
`worldDelta / deltaTime`; gamepad intensity is the normalized stick/trigger
magnitude, keyboard intensity is 1. Defaults are `(0, +Y, 1)` and can be edited
in the Brush 3D widget. Applying the brush synchronizes the fields into the
material's packed GPU block; selecting a different brush does not retroactively
change an already-painted material until it is applied.

The catalogue contains `SweptDirectionalSphereTriangles` and
`SweptDirectionalSphereHexagons`: a tapered capsule from zero radius to the
usual test-object radius, painted with dedicated pattern materials 38/39. The focused
`VULKAN_PATTERN_TEST=1` scene uses the same two swept spheres over a slab and
shows the patterns clearly in debug view 52.

## 4. Normals

The shading normal is

```
N = normalize(grad(F)) = normalize(grad(d_base) - grad(h))
```

`grad(d_base)` is the existing analytic trilinear cell gradient
(`sdfCellNormal`). `grad(h)` is estimated with a regular-tetrahedron finite
difference of the displacement field:

```
grad(h) ~= (3 / 4h) * sum_i (h(p + h e_i) - h(p)) e_i
```

with four unit tetra directions and a sample offset of about half a pixel at
the hit distance (clamped to the cell scale). Four displacement samples give a
stable gradient without an axis-aligned stencil or a six-tap central
difference. The ring's per-vertex base normals are still used for the
undisplaced path (raster parity); with displacement active the true displaced
normal replaces them for lighting, visibility, shadow bias, AO and secondary
rays.

The base normal is recomputed only for the inverse mapping used to recover the
undisplaced shading point; it is not used as the surface normal.

## 5. Depth integration

The fullscreen raymarch writes `gl_FragDepth` from the actual displaced hit
position:

```
clip = viewProjection * vec4(hitPos, 1)
gl_FragDepth = clip.z / clip.w
```

The pass keeps the existing D32 depth attachment, compare op and viewport. It
depth-tests and depth-writes exactly like the old terrain pass, so the displaced
surface occludes and is occluded by rasterized geometry through the same depth
buffer. The water pass, composite, droplets and any depth reconstruction keep
their existing solid-depth dependency. No depth convention was hardcoded; the
projection matrix already carries the engine's clip-space convention.

## 6. Shadow integration

### Self-shadowing / terrain-to-terrain

`solidShadowTrace()` marches the same solid-layer roots from the shaded point
toward the sun. Because it calls the same `rmField()` / `rmRootTestBox()`, the
shadow ray intersects the displaced zero set, not the base surface. The ray
starts at the displaced hit, lifted along the displaced normal by

```
bias = max(0.1, displacementBound * 0.05) + 0.05 * bias * lightDir
```

so a peak cannot shadow its own surface, while the bias scales with the
material amplitude instead of being a fixed world constant.

### Rasterized receivers

`ShadowRenderer` records a fullscreen raymarched terrain caster into every CSM
cascade after the rasterized casters (`OctreeRaymarchShadow.frag`). It
reconstructs the orthographic light ray from the cascade's inverse light-space
matrix, marches the solid roots of the same node table, and writes the
accepted hit's EVSM moments + `gl_FragDepth`. Vegetation, SDF objects, video
layers and any other CSM receiver therefore get shadows from the displaced
terrain without changing their sampling code.

The caster does not need `OctreeDerive.comp`'s payload: it resolves
LAND/NORMAL sentinels itself and reads the corner Lipschitz bound directly, so
it can run before the derive pass in the frame. Its shadow LOD descends while
a cell is larger than two shadow texels.

### Rasterized casters onto displaced terrain

`solidShadowTrace()` also samples the raster CSM (`ShadowCalculation`) and
combines it with the SDF term (`max(csm, sdf)`). The CSM carries every
non-terrain caster, so ordinary geometry shadows displaced terrain. Since the
terrain caster writes the same displaced field, the two terms are consistent
instead of being two different surface approximations.

The global toggle is `Settings::displacementShadowCasterEnabled`; the material
amplitude is the bias scale. It is **opt-in and off by default** because the
three fullscreen cascade marches are too expensive for the full procedural map
on the reference Radeon 680M (they can trip the GPU watchdog at 720p); the SDF
`solidShadowTrace` self-shadowing below is always on and does not require it.
Enable the caster for small scenes or when rasterized receivers specifically
need terrain shadows.

## 7. Ambient occlusion and secondary effects

The material AO texture is unchanged. When AO is enabled and the hit material
uses displacement, a four-probe hemisphere SDF occlusion term is evaluated on
the displaced field and multiplied with the texture AO:

```
occ += clamp((0.35 * radius - F(probe)) / (0.35 * radius), 0, 1)^2
ao  = 1 - 0.5 * occ
```

The probe radius covers `2 * maxAbs` and half a feature wavelength, clamped to
the hit cell. A plane contributes zero because its SDF distance along a tilted
probe is larger than the threshold. This makes displaced peaks/overhangs
darken crevices, not just tint them with the albedo map.

Water reflection and refraction rays (`rmMarchSolidRoots`) use the same field
and the same displaced hit normals as the camera ray, so secondary hits see the
same geometry. The solid environment reflection is procedural sky (there is no
raster scene-color trace for SDF terrain), and that is unchanged.

## 8. Debug views

The Settings debug combo includes:

| View | Shows |
|---|---|
| Original SDF Surface | Base isosurface (displacement bypassed; march and normals use the base field) |
| Displacement Magnitude | Signed `h / bound` (green = outward, red = recessed) |
| Displacement Bounds | Green inside the conservative band, red outside |
| Displaced Normals | `normal(grad(F))` encoded as RGB |
| Displacement March Iterations | Iteration heat of the accepted primary march |
| Displacement Conservative Step | Last accepted step length relative to one pixel |
| Displacement Intersection Residual | `|F|` at the refined hit / hit epsilon |
| Displacement Shadow Ray | R = shadow-ray step heat, G = final occlusion term |

The views are produced by the same march that shaded the hit; they are not a
parallel approximation.

## 9. Performance

The implementation reuses the existing octree table, root sorting, child-cell
clamping and node payload cache. New costs are:

* one or two displacement texture taps per field evaluation (only when the
  material disables legacy mode; legacy was already sampling height maps),
* in-band steps clamped to the material feature wavelength,
* four extra displacement samples at each accepted solid hit for the true
  normal,
* four field probes at each accepted solid hit when AO + displacement are on,
* one fullscreen raymarch per CSM cascade when the terrain caster is enabled.

Outside the displacement band the step is the same base-SDF distance step as
before, so flat/unmapped materials pay nothing. The shadow caster's LOD gate
stops descending at two shadow texels per cell. The new path can be A/B tested
without touching materials through the global flags; the legacy path remains
bit-compatible.

`scripts/bench_raymarch_displacement.sh` runs the deterministic benchmark
camera with displacement off/on and prints the octree march timing from
`logs/run.log`. On the reference AMD Radeon 680M the cost is dominated by the
in-band step count; lowering the amplitude/frequency reduces the band and the
step count. Measured debug-build behaviour (1280x720, deterministic slab +
sphere test scene, geometry filling the view):

| Configuration | Measured `[frame]` |
|---|---|
| Displacement off, deterministic scene | below the app's 100 ms print threshold (≈70–90 ms) |
| amp 1.0, freq 1, AO + SDF shadows | avg 106 ms (100–118) |
| amp 1.0, freq 1, SDF AO off | avg 106 ms (100–133) |
| amp 1.0, freq 1, CSM caster on | avg 124 ms (116–164) |
| amp 0.4, freq 16 | avg 105 ms (100–117) |
| amp 8.0, freq 2, AO + SDF shadows | ~700–1000 ms |
| Full 15 km procedural map, amp 0.5, caster off | ~650–750 ms for the first ready frames (debug build) |

The full procedural map is already close to the Radeon 680M's watchdog budget
even **without** displacement (the latest baseline `logs/run.log` shows
~0.5 s frames followed by a device loss after a long run). Enabling the
three-cascade CSM caster on that map reliably crosses the watchdog, which is
why it is opt-in/off by default. Use the deterministic test scene for
validation and a small amplitude (≤1 m) on the full map. A release build and a
smaller raycast resolution are the next steps for real-time use.

Run the acceptance sweep with:

```
VULKAN_TEST_FRAMES=24 ./scripts/test_raymarch_displacement.sh
```

It covers baseline, positive/negative amplitude, high frequency, offset/bounds,
procedural source, caster off, AO off, the displaced-normal debug view and the
full-map smoke case, and fails on any validation error or device loss.
`VULKAN_DEBUG_MODE=<id>` selects any debug view headlessly.

## 10. Known limitations

* The base field is a trilinear reconstruction, not the exact analytic scene
  SDF. Displacement is added to that reconstruction; a cell with one unknown
  corner (INFINITY sentinel) interpolates over its finite corners, as before.
* The material frequency is a declared band limit. Detail finer than
  `1 / (uvScale * frequency)` and finer than a pixel can be missed by the
  conservative step. The engine cannot prove a bound for arbitrary
  unbounded-height texel content without a per-texel max-slope preprocess;
  the material frequency control is the explicit authoring contract.
* Node material changes are cell-quantized during the march. The shading
  blend and the hit-time ring reconstruct the raster per-vertex material
  transition, but two adjacent cells with very different displacement
  amplitudes can still show a small geometric step at their shared face. The
  same limitation exists in the legacy tessellation path.
* The CSM terrain caster is an extra fullscreen pass per cascade. A scene with
  no vegetation / raster receivers can disable it to recover the cost.
* The secondary-SDF segment is material-level: the latest dragged segment
  orients/animates the whole material. Voxels do not store a segment id, so a
  later stroke reorients earlier stamps of the same material. This is the
  explicit trade-off for keeping the secondary SDF out of the octree.
* Water is never displaced by the solid material height map (the Gerstner
  swell is a shading normal/height offset, not a change of the lake's SDF
  interface); this preserves the existing water/terrain contract.
* The reflection path is environment-only for solid terrain, as before; there
  is no screen-space SDF reflection ray that could show displaced geometry in
  a mirror. Water reflection/refraction rays do traverse the displaced field.

## 11. Future improvements

* Precompute a per-layer maximum height slope (or a mip-pyramid slope bound)
  on the CPU/GPU upload path and feed it into `L_h` for a fully data-derived
  conservative bound instead of the declared-frequency contract.
* Add a per-material displacement LOD / max step clamp to trade precision for
  fill rate on lower-end GPUs.
* Use a shared hierarchical empty-space skip that is displacement-aware (the
  current skip is suspended inside the band, which is correct but not optimal).
* Add a compute-side acceleration structure for shadow rays (the current
  caster re-traverses the node table per shadow texel).
* Extend the displacement texture selector to a full material slot with
  independent invert/offset controls in the editor.

## 12. Files

| File | Role |
|---|---|
| `types/MaterialProperties.hpp` | CPU material controls |
| `vulkan/ssbo/Material.hpp` | Packed GPU material (+3 vec4: displacement/pattern/segment) |
| `vulkan/resources/MaterialManager.cpp` | CPU -> GPU packing |
| `shaders/includes/octree/DisplacementField.glsl` | Shared field/material sampling/wavelength |
| `shaders/renderer/octree/OctreeRaymarch.frag` | Camera march, normals, AO, depth, debug views |
| `shaders/renderer/octree/OctreeRaymarchShadow.frag` | Displaced terrain caster for CSM |
| `vulkan/renderer/octree/OctreeRaymarchRenderer.*` | Caster pipeline + record path |
| `vulkan/renderer/shadow/ShadowRenderer.*` | Per-cascade caster integration |
| `widgets/TextureViewerWidget.cpp` | Material/texture and pattern displacement editor |
| `widgets/Brush3dWidget.cpp` | Secondary-SDF authoring and segment defaults |
| `widgets/RaymarchWidget.cpp` | Global path / AO / caster switches |
| `scripts/test_raymarch_displacement.sh` | Headless A/B + secondary-SDF validation test |
| `scripts/bench_raymarch_displacement.sh` | Timing A/B |

### Explicit pipeline bind in the caster

`OctreeRaymarchRenderer::recordShadowCascade()` is recorded from
`ShadowRenderer`'s per-cascade command buffer. The octree renderer's
`CommandBufferState` pointer belongs to a different producer (solid task), so
the caster binds its pipeline with `vkCmdBindPipeline` directly and updates the
tracker afterwards. Relying on the cached state left the raster EVSM pipeline
active while the caster's descriptor sets were bound, which is
VUID-vkCmdDraw-None-08600.
