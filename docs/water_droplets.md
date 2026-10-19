# GPU water droplets (SDF particle simulation + ray-marched rendering)

A 2048-particle water-droplet system that reuses the engine's octree SDF as
both its collision geometry and its rendering field. The particles are spawned
inside a 128 m sphere at cloud level, fall under gravity, collide with the
terrain SDF, with the water plane and with each other, and are re-seeded every
30 s.

## Reuse: one SDF for rendering and physics

| Piece | Reused component | Notes |
|---|---|---|
| Collision geometry | Octree SDF corner samples | `LocalScene::buildSdfRegionTable` flattens the subtree(s) intersecting the simulation region into `OctreeGpuNode` — the exact node layout the GPU topology pass consumes (`services/OctreeTopologyTypes.hpp`) |
| Node semantics | entry 0 = world root, `type` 1 = surface / 2 = solid / 0 = empty, `flags` bit0 = leaf, corner bit order `bit2=x, bit1=y, bit0=z`, `INFINITY` = no data | identical to `shaders/computation/octree/OctreeTriangles.comp` |
| SDF evaluation | root descent + trilinear corner interpolation | `sdfNodeAt` / `sdfAt` / `sdfGradient` in `shaders/computation/water/DropletSim.comp` mirror the walker's arithmetic (documented duplicate: the topology tables are per-chunk and transient, so a world-space query needs its own persistent table) |
| Particle radius | `Settings::dropletRadius` | the *same* value drives the SDF contact query and the rendered sphere: switching render settings can never change the collision shape |

The region table is built once per scene (never per frame): a rigid-body
transform is just a particle position, so nothing has to be rebuilt when a
droplet moves.

## Simulation (`services/DropletSim.{hpp,cpp}` + `shaders/computation/water/DropletSim.comp`)

One compute pipeline, six passes selected by a push-constant id:

| Pass | Work |
|---|---|
| SPAWN | deterministic hash(i, seed) → uniform point in the spawn sphere, small random velocity, `flags = active` |
| INTEGRATE | semi-implicit Euler: `v.y -= g·dt`, linear damping, `prev = p`, `p += v·dt` |
| GRID CLEAR | zero the 8192-bucket spatial hash counts |
| GRID SCATTER | `atomicAdd` bucket slot per active particle (64 slots/bucket, overflow counted) |
| CONTACTS | one solver iteration: 27-cell pair sweep (half impulses + split positional correction), terrain SDF contact (gradient normal, restitution + Coulomb-clamped friction), water-plane contact |
| FINALIZE | retire particles outside the region, damp resting particles |

* Fixed timestep: the frame delta is split into `Settings::dropletSubsteps`
  equal substeps (clamped to 0.1 s per frame so a hitch cannot tunnel a droplet
  through the SDF in one step).
* `Settings::dropletSolverIterations` iterations of CONTACTS run per substep.
* The solver is Jacobi-style: an invocation reads neighbours but writes only its
  own record, so the result never depends on workgroup order; each side of a
  pair applies half of the shared impulse (momentum conserving).
* All state is device-local; the host never uploads particle data per frame.
  Only the node table (once per scene), the radius/mass pattern (once per
  radius change) and the optional diagnostics counters are host traffic.
* Vulkan: explicit `vkCmdPipelineBarrier2` between passes, plus a final
  compute → vertex/fragment barrier so the droplet draw in the same command
  buffer reads the freshly written state.

## Rendering (`vulkan/renderer/water/DropletRenderer.{hpp,cpp}` + `DropletRenderer.vert/.frag`)

* One camera-facing billboard per droplet (6 non-indexed vertices, instanced
  over the state buffer, camera basis from the inverse view-projection): no
  vertex buffer, no cull-mode guesswork, every covered pixel shaded once.
* The fragment stage sphere-traces the droplet SDF (32 steps, analytic entry as
  the fallback) and shades:
  * refraction through the sphere (`Settings::dropletIor`, entry + exit taps,
    total internal reflection falls back to the reflection sample),
  * reflection along `reflect(v, n)` from the sky view,
  * Schlick Fresnel mixing, feathered silhouette coverage,
  * `gl_FragDepth` from the hit position so later draws depth-test correctly.
* Drawn inside the solid color pass after the opaque geometry: depth test on,
  depth write off, premultiplied coverage blending. Droplets are therefore
  composited with the solid scene and occluded by terrain, water and vegetation
  exactly like other transparent geometry.

## Configuration

`Settings` (see `types/Settings.hpp`) + the *Droplets* widget
(`widgets/DropletsWidget.{hpp,cpp}`): master toggle, count, spawn radius/height,
particle radius, reset interval, gravity, restitution, friction, damping, IOR,
solver iterations, substeps, stats readback, "Reset now".

Headless / diagnostics environment variables:

| Variable | Effect |
|---|---|
| `VULKAN_DROPLETS=1` | enable droplets + stats (same state as the widget) |
| `VULKAN_DROPLET_COUNT=N` | particle count (1..8192) |
| `VULKAN_DROPLET_RADIUS=R` | particle radius in metres |
| `VULKAN_DROPLET_DT=dt` | force a fixed per-frame simulation delta (slow software rasterizers) |
| `VULKAN_STREAM_DIAG=1` | chunk streaming counters (see AGENTS) |

`Settings::dropletStats` (or `VULKAN_DROPLETS=1`) prints one line per second:

```
[droplets] pairContacts=.. terrain+escaped=.. gridOverflow=.. maxContactsPerParticle=..
           | active=.. y[min=.. mean=.. max=..] | p0=(x,y,z) r=.. |v0|=..
```

`terrain+escaped` counts SDF contacts **and** particles retired by the finalize
pass (the shader has one shared counter slot for both); `gridOverflow` counts
particles dropped from a full hash bucket.

## Known limitations / next steps

* **Contact generation is sphere-based.** A droplet is a sphere SDF, so the
  narrow phase is an exact sphere-vs-SDF query (distance + gradient). Arbitrary
  rigid-body SDF contacts (multi-contact manifolds between extended bodies,
  nonuniform scaling, concave shapes) are *not* implemented: that needs
  surface sampling or iterative closest-point manifolds per pair.
* **No continuous collision detection.** The terrain is a volume SDF, not a thin
  shell, so a fast droplet that overshoots the surface is pushed back out over
  the next iterations instead of tunnelling permanently; thin geometry would
  need conservative advancement or swept bounds.
* **No warm starting / persistent contacts** (contacts are rebuilt every
  substep), so resting stacks are stable but not perfectly rigid.
* **Rendering has a single mode** for droplets (direct SDF ray marching). The
  generic per-object `renderMode` / `maxLod` / `pixelErrorThreshold` /
  `materialLodBias` configuration of the wider design is not implemented: the
  adaptive-mesh path for these particles would mean generating 2048 sphere
  meshes per frame, which the ray-marched path exists to avoid.
* **Water-plane contact** uses the flat level (y = 0, the water box top from
  `MainSceneLoader`); the water layer's own SDF is not queried.
