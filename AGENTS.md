# Vulkan Engine - AGENTS.md

## Build & Run

| Command | Action |
|---------|--------|
| `make` / `make all` | Release build (app + server + shaders, ImGui enabled, `-O3 -march=native`) |
| `make debug` | Debug build (`-O0 -g -DDEBUG`, no ImGui) |
| `make run` | Release build + run `bin/app` (CWD is `bin/`) |
| `make run-debug` | Debug build + run |
| `make shaders` | Compile `shaders/*.{vert,frag,geom,comp,tesc,tese}` → `bin/shaders/*.spv` via `glslc` (fallback `glslangValidator`) |
| `make server` | Build headless `bin/server` (no Vulkan/UI linkage, no widgets) |
| `make clean` | Remove `bin/` and generated SPIR-V |
| `make valgrind` | Debug build + valgrind with `valgrind.supp` |
| `make install` | Install system deps + fetch ImGui + miniaudio into `third_party/` |
| `make cloc` | Count lines of code (excludes `bin/`, `third_party/`) |
| `make callgrind` | Profiling with valgrind callgrind + kcachegrind |
| `MAKE_JOBS=16 make` | Override parallel jobs (default 8) |

Compiler is `g++` (`-std=c++23 -pthread`). Dependencies via `pkg-config`: glfw3, vulkan, stb, jpeg, gdal, zlib. Optional Wii Nunchuk via libcwiid (auto-detected). Shaders target `vulkan1.1`. Build outputs go to `bin/` and the binary expects resources (textures, fonts, shaders) relative to CWD — always run from `bin/`. ImGui is only compiled with `-DUSE_IMGUI` (release builds). Never modify `third_party/` files.

## Architecture

Two entry points: `main.cpp` (class `MyApp` extends `VulkanApp`) and `server.cpp` (headless).

- `vulkan/` — Vulkan setup, resource management, renderers (`vulkan/renderer/`)
- `vulkan/renderer/` — SceneRenderer (orchestrator), SolidRenderer, SkyRenderer, WaterRenderer (tessellation), ShadowRenderer (cascaded shadow maps), VegetationRenderer (GPU), IndirectRenderer (GPU frustum culling), PostProcessRenderer, ImpostorCapture, DebugCubeRenderer, DebugSDFRenderer, WireframeRenderer, WaterBackFaceRenderer, Solid360Renderer, CubeToEquirectRenderer
- `vulkan/ubo/` — GPU uniform buffer structs; `vulkan/includes/` — shared C++ headers (locations.hpp, vertex_layouts.hpp)
- `space/` — Octree, Tesselator (Surface Nets meshing), ThreadPool, ConcurrentQueue, Processor, Simplifier, OctreeVisibilityChecker
- `sdf/` — SDF primitives (Box, Sphere, Capsule, Cylinder, Cone, Torus, HeightMap via GDAL, RoadDistanceFunction, TriangleStrip, OctreeDifferenceFunction, Wrapped* variants for composition)
- `events/` — Input system (keyboard, gamepad, nunchuk), EventManager, IEventHandler
- `math/` — Camera, Light, BoundingBox, Frustum, Plane, Ray, HeightMap, HeightMapTif, PerlinSurface, Transformation, Brush3d
- `services/` — TextureMixer (compute texture blending), BillboardService, ImpostorService
- `widgets/` — ImGui debug/editor UI (compiled only with `-DUSE_IMGUI`, i.e. release builds)
- `tree/` — AttractorField, TreeGenerator, TreeHandler
- `utils/` — LocalScene, MainSceneLoader, FileReader, SettingsFile, brush system (LandBrush, WaterBrush, Brush3d)

Three async queues (graphics, vegetation compute, geometry compute) each with own `VkCommandPool`. Uses `deferDestroyUntilFence()` for GPU-safe object cleanup. Frames in flight: 3. StagingRingBuffer for uploads.

## Vulkan Rules

Read `.github/copilot-instructions.md` for the full set (553 lines). Critical constraints:

- **Synchronization2** everywhere (`VkImageMemoryBarrier2`, `vkCmdPipelineBarrier2`), no legacy barriers
- **Zero validation warnings/errors** tolerated at any time — they are the test suite
- No `vkDeviceWaitIdle` / `vkQueueWaitIdle` in the render loop or streaming path
- No per-chunk Vulkan buffer allocation; use shared global buffer pools
- Texture arrays: `VK_IMAGE_VIEW_TYPE_2D_ARRAY` with per-subresource (layer/mip) layout tracking
- Timeline semaphores preferred for GPU-GPU sync; fences only for GPU-CPU (resource retirement, frames in flight)
- Compute mixer: storage images in `GENERAL`, sampled in `SHADER_READ_ONLY_OPTIMAL`, never read/write same layer in same dispatch
- Geometry streaming: worker threads generate mesh data (no Vulkan access), staging ring buffers for upload, device-local vertex/index buffers, never block CPU on uploads
- "One way to solve a problem, the best way" — no fallbacks. If a fallback is better, it replaces the main approach.

## No Testing Framework

No unit tests. Validation layers + valgrind + ImGui debug widgets are the QA strategy.
