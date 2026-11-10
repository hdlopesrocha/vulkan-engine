# GPU virtual adaptive refinement for Dual Contouring

Status: implemented in `shaders/computation/octree/OctreeVirtualRefine.comp`,
dispatched by `services/OctreeTopologyCompute` before
`shaders/computation/octree/OctreeTriangles.comp`. The octree, its node
allocation, the flattening in `utils/LocalScene.cpp`, and the packed vertex /
index pools are unchanged.

## 1. What is refined

The GPU does not have an analytic evaluator for the scene's arbitrary
`SignedDistanceFunction` graph. A node carries exactly its eight corner SDF
samples. The field available to the mesher is therefore the trilinear
reconstruction of those corners, which is also the field the existing edge
walker (`SDF::interpolate`) and the CPU QEF (`SDF::getPosition`) assume.

Virtual cells sample that reconstruction. They never allocate an octree node,
never touch the host, and never persist: the queue lives in the invocation's
private scratch.

```
level 0    the node's 12 edges -> baseline QEF
level d    subdivide selected level-(d-1) cells into 8 children
           add every child's owned edge intersections to the running QEF
           rank children and select the next frontier
```

The 3x3x3 half-resolution lattice reuses the 8 parent corners and evaluates
only the 19 new samples per subdivided parent. Sibling cells share their face
and edge samples, and the lattice is evaluated once per parent, not once per
child.

## 2. Intersection detection

For every cell edge with finite endpoints and opposite signs:

```
t = clamp(d0 / (d0 - d1), 0, 1)
p = (1 - t) p0 + t p1
n = normalized trilinear gradient at p
```

A zero-valued corner is classified as non-negative (the existing rule), so an
edge to a negative neighbour produces `t = 0`; coincident `t = 0` samples from
the incident edges are deduplicated per cell. A zero at one of the original
node's corners keeps the existing corner-snap path and is never refined, so
shared cells keep bit-identical positions.

Normals use the existing `SDF::getNormalFromPosition` math. For the refined
QEF only, the gradient is blended with the edge direction when the field's
derivative along the edge is near 1:

```
slope = |d1 - d0| / edgeLength
s = clamp((slope - featureSlope) / (1 - featureSlope), 0, 1)
n = normalize((1 - s) * gradientNormal + s * edgeDirection)
```

This is not corner/edge/face classification of the *priority*: it is a
per-sample normal estimate. A sharp, locally planar face makes the field change
at unit rate along the crossing edge while the trilinear gradient smears the
corner; the edge direction recovers the face normal. On a smooth surface the
slope is below the threshold and the existing gradient is used unchanged.
`VULKAN_REFINE_SLOPE=1` disables the blend and restores gradient-only normals.

## 3. Edge ownership and QEF weighting

Four cells share an interior lattice edge. The owner is the cell with the edge
on its minimum face in both perpendicular axes; on the domain's maximum
boundary the last cell takes it. Each unique lattice edge therefore contributes
exactly once per level. Without this, shared edges would be weighted 2x or 4x
and bias the vertex toward cell centres.

The refined QEF accumulates every level's owned intersections:

```
A = sum n n^T,  b = sum n (n . p),  c = sum (n . p)^2
```

and solves with the same `0.1` mass-point regularizer and Cramer solver as
`SDF::getPosition`, then clamps to the original node cube. Refinement cannot
produce a vertex outside its spatial region.

## 4. Priority and selection

For a cell, using gradient normals:

* `density = count / 6`
* `variation = 1 - |mean(n)|`
* `residual = sqrt((x^T A x - 2 x^T b + c) / count) / cellSize`
* `distance = |f(x)| / (|grad f(x)| * cellSize)`
* `dispersion = rms(intersection - centroid) / cellSize`

```
priority = 0.30 density + 0.30 min(2 variation, 1)
         + 0.20 min(4 residual, 1) + 0.15 min(2 distance, 1)
         + 0.05 min(dispersion, 1)
```

Density is deliberately only 30%. A plane has many crossings (up to 6) but zero
normal variation, residual and surface distance; a sharp corner has only 3
crossings and a large variation/residual, so it wins. Counting intersections
alone would do the opposite.

Selection is a stable top-`budget` insertion sort keyed by
`(bestChildOfParent, priority, cell key)`. The best child of every parent is
kept before the remaining budget is filled by priority. Every intersected cell,
selected or not, still contributes its owned edge samples to the running QEF;
selection only decides where the next level samples. A second, lower-priority
surface component is therefore never dropped from the vertex fit.

## 5. Stopping and fallbacks

* Level 0 is skipped when the coarse QEF is already on the sampled zero set
  (`distance <= tolerance/2`) or is an exact local plane
  (`variation < 0.01 && residual < 1e-5 && distance < 1e-3`).
* After each level, stop when the normalized surface error and the position
  change are both below `tolerance`, or when the position is stable
  (`delta <= tolerance/4`).
* `maxDepth` (default 2, hard cap 3) and `budget` (default 8, hard cap 8) bound
  the work.
* Sentinel corners (`INFINITY`), exact-zero node corners, non-surface nodes and
  nodes without a vertex slot write `w = 0`; `OctreeTriangles.comp` then uses
  the exact `sdfCellPosition()` result. The same happens for every node when
  `VULKAN_VIRTUAL_REFINE=0`.

A final gradient projection onto the sampled zero set is available with
`VULKAN_REFINE_PROJECT=1` but is **off by default**: it lowers the absolute
trilinear SDF value while rounding sharp features, which is the opposite of the
target.

## 6. GPU architecture

### Dispatch order and synchronization

```
vkCmdFillBuffer(nodeWritten)          transfer
  -> transfer/compute barrier
vkCmdDispatch(OctreeVirtualRefine)    one invocation per node
  -> compute/compute buffer barrier   refinedPos[] write -> read
vkCmdDispatch(OctreeTriangles)        one invocation per scan cell
  -> pool barriers, host-read barrier
```

The refine pass writes one `vec4` per node (`xyz` = position, `w` = usable
flag). The geometry pass reads that entry in `cellPos()` and falls back to
`sdfCellPosition()` when `w` is 0. The interface is a single compute ->
compute buffer barrier; there is no host readback between levels.

The optional diagnostic timestamp query is created per chunk only when
`VULKAN_SETUP_DIAG` is set and the device reports timestamp support; it wraps
the refine dispatch only.

### Persistent memory

| Resource | Size | Notes |
|---|---:|---|
| `bRefinedPos` | 16 B / node | device-local, written by refine, read by geometry |
| `bRefineStats` | 64 B | host-visible diagnostic counters |
| global virtual-cell arena | 0 B | none; the queue is invocation-private |
| scratch / invocation | ~3.5 KiB worst case | `8` current + `64` next cells x 48 B |

No buffer grows with `depth^3`; increasing depth or budget only changes the
bounded local queue. Input node tables are the existing ones.

### Determinism

One invocation owns one node. Children are visited in fixed order, ownership is
a pure function of `(level, ijk)`, selection has a key tie-break, and no
cross-invocation state is used. The same node sampled by two neighbouring
chunks therefore produces the same refined position bits, preserving the
shared-edge stitching.

## 7. Diagnostics

`VULKAN_SETUP_DIAG=1` prints, every 256 chunks and in the final summary:

* `nodes` — slot-bearing surface nodes processed
* `refined` — nodes that ran at least one level
* `cells` — virtual child cells generated
* `evals` — trilinear corner evaluations
* `inter` — surface intersections detected
* `selected` — cells selected for a deeper level
* `depth` — maximum depth reached
* `resBase` / `resRef` — normalized QEF residual before/after
* `sdfBase` / `sdfRef` — normalized distance to the sampled zero set before/after
* `absSdf` — normalized final `|f(vertex)|`
* `gpu` — timestamped refine-dispatch time, microseconds/chunk
* `skipAcc` / `skipUnsup` / `overflow`

Fixed-point sums use a `1e-3` scale to keep the `uint` accumulators bounded.

## 8. Configuration

| Environment variable | Default | Range | Effect |
|---|---:|---:|---|
| `VULKAN_VIRTUAL_REFINE` | `1` | 0/1 | master switch; `0` is the exact pre-refinement path |
| `VULKAN_REFINE_DEPTH` | `2` | 0..3 | maximum virtual depth |
| `VULKAN_REFINE_BUDGET` | `8` | 1..8 | cells kept for the next level |
| `VULKAN_REFINE_MIN_PRIORITY` | `0.35` | 0..1 | deeper-level selection threshold |
| `VULKAN_REFINE_SLOPE` | `0.90` | 0..1.0 | edge-normal blend threshold; 1.0 = gradient only |
| `VULKAN_REFINE_TOL` | `0.01` | 1e-5..0.5 | normalized geometric tolerance |
| `VULKAN_REFINE_PROJECT` | `0` | 0/1 | optional final zero-set projection |

The pass can also be toggled at runtime with
`OctreeTopologyCompute::setVirtualRefinementEnabled(bool)`.

## 9. Validation

### Offline model

A standalone Python model uses the same corner samples, trilinear field, edge
ownership, priority and QEF as the shader. Four hundred random cells per case;
error is the distance from the produced vertex to the analytic surface,
normalized by cell size (Newton projection to the analytic zero set).

| Case | Cells | Before | After | 95th before | 95th after | Improved | Avg. virtual cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plane | 304 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — | 1.0 |
| Sphere | 399 | 0.2870 | 0.2706 | 0.5317 | 0.5127 | 41% | 8.5 |
| Sharp cube | 399 | 0.2523 | 0.2077 | 0.4372 | 0.4558 | 50% | 8.5 |
| Rounded cube | 393 | 0.2609 | 0.2006 | 0.3944 | 0.3935 | 61% | 10.3 |
| Two planes / wedge | 400 | 0.1222 | 0.0327 | 0.1732 | 0.0979 | 83% | 7.6 |
| Terrain height field | 277 | 0.0858 | 0.0858 | 0.1674 | 0.1674 | — | 1.1 |
| Two nearby spheres | 164 | 0.1095 | 0.1043 | 0.2483 | 0.2483 | 12% | 2.3 |

Planar and already-accurate terrain cells are skipped before level 1; the
remaining error is the analytic field's difference from the stored samples,
which no virtual cell can remove.

### Actual shader run (lavapipe compute harness)

The real SPIR-V was executed in a standalone Vulkan compute harness with
synthetic node tables; the baseline is the existing `sdfCellPosition` QEF and
the error is again the true analytic distance.

Mixed 205-node set (planes, spheres, boxes, corners, wedges, exact-zero
fallbacks):

| Metric | Baseline | GPU refined |
|---|---:|---:|
| Mean normalized error | 0.1618 | **0.1263** (−22%) |
| Nodes improved | — | 43% |
| Refined nodes | — | 119 / 205 |
| Skipped as accurate | — | 79 |
| Unsupported / fallback | — | 7 |

Targeted sharp-corner set (199 cells each containing a box corner):

| Metric | Baseline | GPU refined |
|---|---:|---:|
| Mean normalized error | 0.2076 | **0.1158** (−44%) |
| Median | 0.1987 | **0.0864** (−57%) |
| 95th percentile | 0.3193 | 0.3122 |
| Nodes improved | — | 63% |

The mixed run produced 1240 virtual cells, 2945 trilinear evaluations, 3563
intersections, 36 selected cells (next-level work only), and maximum depth 2. Plane and exact-zero
corner nodes fell back bit-exactly and were unchanged.

The offline model and the shader agree in aggregate (0.1246 vs 0.1263 mean on
the mixed set). They differ in 7 borderline selections because the model is
float64 and the shader is float32; priority is a continuous score and a cell
that sits on `minPriority` can switch sides. Lowering `VULKAN_REFINE_MIN_PRIORITY`
makes the selection more inclusive; raising it makes it more conservative.

### What can mislead the priority

* **Intersection count alone.** A plane crossing a cube can produce 6
  intersections; a sharp box corner produces 3. A count-only priority refines
  planes more than corners. The 30% density term plus variation/residual/
  distance terms is what prevents this.
* **Trilinear information limit.** The GPU cannot recover detail absent from
  the eight corner samples. Refining the trilinear field *without* the
  edge-normal blend lowers the sampled-field error but can move a 90-degree
  corner farther from the true surface; that is why the blend exists and why
  projection is off by default. On the 199-cell sharp-corner set the
  gradient-only shader moved the mean true error from 0.2076 to 0.2132 (only
  15% of cells improved) while *reducing* the sampled-field error; the
  edge-normal blend moved it to 0.1158 (63% improved). A metric based only on
  `|f(vertex)|` would have selected the worse behavior.
* **Disconnected components in one cell.** All intersections are kept and all
  contribute to the QEF, but one cell still produces one vertex. If two
  disjoint surfaces require different vertices, no single QEF point can satisfy
  both; the refined vertex is a compromise, as is the baseline. The
  best-child-per-parent rule ensures every selected region continues to be
  sampled, so a component is not silently dropped.
* **Thin slivers / tangent contacts.** A cell can have high normal variation
  from one or two nearly tangent crossings. The residual and surface-distance
  terms keep these low, and single-crossing cells are below the selection
  threshold in practice.
* **Threshold sensitivity.** Priorities are continuous; a cell near
  `minPriority` may be selected on one vendor and not another. The aggregate
  accuracy is stable, but per-vertex bits are not expected to match a float64
  reference when the refinement is enabled.

## 10. Integration summary

| File | Change |
|---|---|
| `shaders/computation/octree/OctreeVirtualRefine.comp` | new bounded per-node virtual refinement pass |
| `shaders/computation/octree/OctreeTriangles.comp` | reads `refinedPos[]` in `cellPos()` when enabled |
| `services/OctreeTopologyCompute.{hpp,cpp}` | pipeline, buffers, descriptor bindings 12/13, dispatch/barrier, env settings, diagnostics |
| `docs/virtual_refinement.md` | this report |

The CPU/offline exporter path (`sdf/SDF.cpp`, `space/OctreeTriangleTessellator`)
is untouched. With `VULKAN_VIRTUAL_REFINE=0` the runtime geometry pass is the
previous implementation, including its bit-exact CPU parity.
