# Video Textures

Video clips become **animated materials**. Every clip owns one layer of the
engine's dedicated **video texture array**; materials of kind `Video` sample
that array through their `textureIndex` (the video slot), so several materials
can reference the same clip and clips can be loaded/unloaded at runtime without
touching the material texture arrays.

The source file stays compressed; frames are decoded on demand for the timestamp
the playback clock asks for, converted to RGBA8 on the GPU and written into the
clip's video-array slot.

```
videos/base/*.mp4 ──(make videos)──> videos/*.mp4   (square, short GOP)
                                          │
  material: kind=Video, textureIndex=N ───┤
                                          ▼
  worker thread (per clip)          main thread (per frame)              GPU
  ┌─────────────────────┐           ┌──────────────────────┐      ┌──────────────────┐
  │ demux + decode      │  publish  │ clocks, request the  │ copy │ packed YUV image │
  │ (vulkan/vaapi/sw)   │──────────>│ newest frame, submit │─────>│ (R8, 1.5 B/px)   │
  │ seek = keyframe +   │   slot    │ the upload (never    │      │        │         │
  │ decode to timestamp │           │ blocks)              │      │ compute shader   │
  └─────────────────────┘           └──────────────────────┘      │ video array slot │
                                                                  └──────────────────┘
                                                                           │
                       video layer pass (after solid + water) samples it ──┘
```

## Assets

`videos/base/` holds the 3 shortest clips, is committed, and is loaded by
`MyApp` at setup. `videos/extra/` is the optional local library (ignored by git
because it can be very large); drop your own clips there.

`make videos` converts `videos/base/*` into `videos/<name>.mp4` (flat, next to
the source folders), and `make videos-extra` converts `videos/extra/*` the same
way. The converted files are gitignored; the normal build copies them into
`bin/videos/` next to the executable.

| Setting | Value | Why |
|---|---|---|
| size | 1024×1024 (square center crop) | matches the default texture-array size, so a clip maps 1:1 onto a video-array slot |
| codec | H.264, `-crf 26 -preset veryfast` | small files, hardware-decodable everywhere |
| GOP | `-g 30 -keyint_min 15 -sc_threshold 0` | short keyframe interval keeps seek latency bounded (≈0.5 s per seek at 30 fps) |
| pixel format | `yuv420p` 8-bit | the conversion shader assumes 4:2:0 8-bit input |
| audio | dropped (`-an`) | not sampled |

At runtime `VideoTextureService::listAvailableVideos()` discovers clips from
`videos/base` and `videos/extra` (next to the executable or in the repository
root) and prefers the converted `videos/<name>.mp4` when present. When only the
converted clips are reachable (running from `bin/`), every converted clip is
exposed as a base clip. `VideoTextureService::listVideoFiles` and
`findAssetDirectory` remain available for simple directory scans.

Environment switches:

| Switch | Effect |
|---|---|
| `VULKAN_NO_VIDEO=1` | do not create the video service/array |
| `VULKAN_VIDEO_SLOTS=<n>` | video array capacity (default 8, max 32) |
| `VULKAN_VIDEO_AUTOPLAY=<n\|all>` | clips that start playing (default 4); paused clips still materialize the frame at their clock |
| `VULKAN_VIDEO_DEBUG=1` | diagnostics every 5 s |
| `VULKAN_VIDEO_BENCH=1` | decode/seek benchmark at startup |
| `VULKAN_VIDEO_BACKEND=auto\|vulkan\|vaapi\|software` | force a decode backend |

Memory: the video-specific cost per clip is one packed YUV image (≈1.5 MiB at
1024²) plus two host-visible staging buffers of the same size. The video array
is `slots × RGBA8 + mips` shared by all clips (≈5.3 MiB per slot at 1024² with
a full mip chain).

## Materials

`MaterialProperties` (mirrored by the GPU `Material` SSBO) gained:

| Field | Meaning |
|---|---|
| `kind` | `Texture` (scene arrays), `Video` (video array slot), `Water` |
| `textureIndex` | layer/slot the material samples; `-1` = the material's own layer |
| `textureMapping` | `Triplanar`, `ProjectionX/Y/Z`, `SdfGiven`, `Custom` |
| `mappingOrigin` | `SdfGiven` sphere: xyz center, w radius |
| `blendMode` | `Normal` + 19 Photoshop-style modes |

`SdfGiven` maps the world position into spherical coordinates around the
material's bounding volume (`center`, `radius`), so an SDF object can carry a
spherical projection. `Custom` uses the mesh/painted UVs. Every technique is
implemented in `shaders/includes/TextureMapping.glsl` and used by the solid,
displacement and video-layer shaders.

All shapes in `MainSceneLoader` keep triplanar mapping. The extra demo cube in
the video layer uses the first video material: `SdfGiven` coordinates centered
on the cube and `Screen` blending.

The Materials widget (Textures panel) edits the material kind, the video slot,
the coordinate technique (with the SDF center/radius) and the blend mode.

### Blend modes

`shaders/includes/BlendModes.glsl` implements, in shader, the 19 modes selectable
per material: Darken, Multiply, Color Burn, Linear Burn, Darker Color, Lighten,
Screen, Color Dodge, Lighter Color, Overlay, Difference, Exclusion, Subtract,
Divide, Hue, Saturation, Color, Luminosity (plus Normal).

Because a forward draw cannot read the destination it blends with, materials of
kind `Video` are rendered by the **video layer** (see below) into their own
target and composited by the post-process pass, where every mode is applied
against the finished scene color. The frame therefore composites as
**Solid → Water → Video**.

The blend-mode field is editable on every material (texture/water/video kinds),
but the shader-side mode application happens in the video-layer composite: the
solid and water passes write opaque/alpha-blended coverage without reading the
destination, so their materials keep the layer's fixed compositing. Use a Video
material (any material can be switched to kind Video in the Materials widget,
including several materials sharing one clip slot) when a Photoshop-style blend
mode is needed.

## Video layer

`vulkan/renderer/video/VideoRenderer` owns a color+depth target per frame and
draws the scene's `LayerRendererType::Video` layer (an octree layer populated by
`MainSceneLoader`; layer 2 of `LocalScene`). It reuses the engine's octree mesh
format and the non-tessellated solid vertex shader; the fragment shader samples
the video array at the material's slot and writes:

- `rgb` = the video color (linear, like the albedo arrays),
- `a` = `(blendMode + 1) / 32`, with `0` meaning "no video" (cleared pixels).

The post-process composite (`PostProcessRenderer.frag`) decodes the mode,
occludes the video fragment against the solid/vegetation depth, and applies the
blend mode. Water is intentionally composited before the video layer.

The layer is drawn on the main command buffer after all async passes have been
submitted (their timeline waits are part of the main submit), so it sees the
uploaded frames of the current frame and the frame's scene uniforms.

## Decoding backend

The backend is probed per clip, best first:

1. **Vulkan Video** — libavcodec's `vulkan` hwaccel
   (`VK_KHR_video_decode_queue` + the codec's decode extension).
2. **VAAPI** — the first `/dev/dri/renderD*` node.
3. **software** — libavcodec's CPU decoders.

Device creation is not proof that a codec can decode with a backend (Vulkan
Video advertises `AV_PIX_FMT_VULKAN` even when the decode extension is missing,
and libavcodec silently falls back to software), so the negotiated pixel format
confirms the backend per clip. A failed candidate is dropped for the session and
the next one is tried; the result is reported in `VideoState::backend` and logged
once per clip.

## Decoding and seeking

* A clip is never fully decoded: only the codec's reference frames and the
  packed YUV image of the last published frame exist.
* The clock advances by `frameDelta × rate`. A newer timestamp request
  supersedes an older one (`requestSeq`), so a stalled decoder never works
  through a backlog.
* A seek maps to the closest preceding keyframe (`AVSEEK_FLAG_BACKWARD` +
  `avcodec_flush_buffers`) and decodes forward to the requested timestamp;
  forward playback needs no seek.
* Requests for an already materialized timestamp are answered from the
  published frame (cache hits).
* VFR is handled through `best_effort_timestamp`: the materialized timestamp,
  not the clock, drives the next staleness test.
* While a frame is not ready the material keeps sampling the previous one — the
  render loop never waits.

## Synchronization and lifetimes

Uploads are recorded on the main thread and submitted through
`submitCommandBufferAsyncToQueue` on the graphics queue:

* **write-before-read** — the submit signals a binary semaphore registered in
  `m_extraWaitSemaphores`; the next frame waits on it before recording/sampling.
* **read-before-write** — the submit waits on the frame timeline at the last
  submitted value, so the conversion cannot overlap a frame that still samples
  the slot.
* The staging slot and the slot's tracked layout are released from a
  `deferDestroyUntilFence` callback holding a weak reference to the service (a
  no-op after `shutdown()`).
* The command buffer transitions the packed image
  `SHADER_READ_ONLY → TRANSFER_DST → SHADER_READ_ONLY` and the video-array slot
  `SHADER_READ_ONLY → GENERAL → TRANSFER_DST → (mips) → SHADER_READ_ONLY`.
* Unloaded clips keep their GPU objects until shutdown (they are registered with
  the app's resource manager), so no deferred callback can observe a dangling
  slot; the slot itself is cleared to black on unload.

## Playback control

The **Video Textures** widget lists the loadable clips (`Load`/`Unload`), the
loaded clips with their slot, backend, codec, resolution, clock, instrumentation,
and offers play/pause, loop, rate, seek, "Play all"/"Pause all", a shared clock
for synchronizing several clips, and "Log diagnostics". The same operations exist
on the service (`setPlaying`, `togglePlaying`, `setPlayingAll`, `setLooping`,
`setRate`, `seek`, `seekAll`, `loadVideo`, `unloadVideo`).

## Diagnostics

* `VULKAN_VIDEO_DEBUG=1` — per-clip line + total every 5 s: backend, clock,
  decoded/uploaded frames, seeks, cache hits, average decode/seek latency, last
  upload CPU time, GPU/staging bytes.
* `VideoTextureService::logDiagnostics()` — the same dump on demand, including
  the comparison against a naive all-frames-resident RGBA implementation.
* `VULKAN_VIDEO_BENCH=1` — sequential decode throughput, random seek+decode
  latency, resident bytes vs. a fully decoded copy.

### Measured on the development machine

AMD Radeon 680M (RADV), FFmpeg 6.1.1, 1024² H.264 clips from `make videos`:

| Clip | sequential decode | random seek + decode | resident video bytes | naive all-frames RGBA |
|---|---|---|---|---|
| 60 s @ 30 fps | 173 fps (5.8 ms/frame) | 112 ms | 4.5 MiB | 7200 MiB (0.06 %) |
| 10 s @ 24 fps | 183 fps (5.5 ms/frame) | 92 ms | 4.5 MiB | 960 MiB (0.47 %) |

The seek latency is dominated by the keyframe interval (a seek decodes at most
one keyframe-to-target run, ≈1 s of video at a 30-frame GOP).

## Build requirement

The backend is optional: the Makefile enables it when
`pkg-config --exists libavformat libavcodec libavutil libswscale` succeeds
(`-DHAVE_FFMPEG`). Without it the engine builds and runs unchanged;
`VideoTextureService::backendCompiledIn()` returns false and the log says video
textures are unavailable.
