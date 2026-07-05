# Vulkan Engine - AGENTS.md

## Project Overview

This project uses **modern Vulkan** and targets **Vulkan 1.4** when available, while maintaining compatibility with Vulkan 1.3 whenever practical through runtime feature detection. Do not assume that every GPU supports Vulkan 1.4 features. Always query feature support before using any optional functionality. :contentReference[oaicite:0]{index=0}

The codebase is written in modern C++ (C++20 or newer) and prioritizes:

- Cross-vendor compatibility
- Explicit synchronization
- High performance
- Readability
- Validation-clean execution
- Low CPU overhead

---

## Build & Run

| Command | Action |
|---------|--------|
| `make` / `make all` | Release build (app + server + shaders, ImGui enabled, `-O3 -march=native`) |
| `make debug` | Debug build (`-O0 -g -DDEBUG`, no ImGui) |
| `make run` | Release build + run `bin/app` (CWD is `bin/`, never close the app or use timeouts, the program is always closed by the user) |
| `make run-debug` | Debug build + run | (Never close the app or use timeouts, the program is always closed by the user)
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

## General Vulkan Guidelines

Always prefer modern Vulkan APIs over deprecated or legacy approaches.

### Prefer

- Vulkan 1.4 core functionality when supported
- Dynamic Rendering
- Synchronization2
- Timeline Semaphores
- Descriptor Indexing
- Buffer Device Address
- Scalar Block Layout
- Dynamic State whenever appropriate
- SPIR-V 1.6
- VK_KHR_maintenance extensions that have been promoted to core

Avoid introducing legacy render passes unless required for backwards compatibility. :contentReference[oaicite:1]{index=1}

---

## Feature Detection

Never assume support based only on the Vulkan version.

Always:

- Query instance version
- Query physical device version
- Query individual features
- Query extensions
- Gracefully fall back when features are unavailable

Feature detection is preferred over version checks. :contentReference[oaicite:2]{index=2}

---

## Validation

Development builds must always enable:

- Khronos Validation Layers
- GPU-assisted validation when practical
- Synchronization validation
- Best Practices validation

Code submitted with validation errors is considered incorrect.

Never ignore validation messages.

Validation warnings should be investigated before merging. :contentReference[oaicite:3]{index=3}

---

## Synchronization

Always use Synchronization2 APIs.

Preferred objects:

- vkCmdPipelineBarrier2
- vkQueueSubmit2
- VkDependencyInfo

Avoid legacy synchronization APIs unless maintaining old code.

Document every barrier with a short explanation describing why it exists.

---

## Dynamic Rendering

Prefer Dynamic Rendering over traditional Render Passes.

Avoid creating render passes unless required by compatibility code.

Use rendering attachments directly.

---

## Memory

Prefer:

- Dedicated allocations only when beneficial
- Large suballocated heaps
- Buffer Device Address
- Persistent mapped upload buffers
- Staging buffers for device-local resources

Avoid unnecessary allocations.

Minimize vkAllocateMemory calls.

---

## Buffers

Prefer:

- Large shared buffers
- Offsets instead of many small buffers
- Device-local memory for GPU resources
- Host-visible staging memory only for uploads

Avoid creating one buffer per object.

---

## Images

Prefer:

- Immutable textures
- Correct layouts
- Explicit ownership transitions
- Optimal tiling

Avoid unnecessary layout transitions.

---

## Descriptors

Prefer:

- Descriptor Indexing
- Descriptor Arrays
- Bindless techniques when appropriate
- Descriptor Buffers if supported

Avoid creating many descriptor sets every frame.

Reuse descriptor resources whenever possible.

---

## Command Buffers

Prefer:

- Secondary command buffers only when beneficial
- Command buffer reuse
- Command pools per thread
- Reset instead of recreate

Avoid excessive command buffer allocation.

---

## Multithreading

Recording command buffers from multiple threads is encouraged.

Each worker thread should own:

- Command pool
- Temporary allocators
- Scratch resources

Avoid global mutable state.

---

## Pipeline Creation

Prefer:

- Graphics Pipeline Libraries
- Pipeline Cache
- Pipeline compilation during loading screens
- Pipeline reuse

Avoid runtime pipeline compilation during gameplay.

---

## Shader Guidelines

Use:

- GLSL or HLSL compiled through glslang or DXC
- SPIR-V 1.6
- Explicit layouts
- Scalar layouts where appropriate

Shaders should avoid:

- Undefined behavior
- Implicit conversions
- Vendor-specific behavior

---

## Performance

Always consider:

- CPU overhead
- Memory bandwidth
- Pipeline state changes
- Descriptor updates
- Resource lifetime

Prefer batching over many tiny draw calls.

Minimize synchronization.

Avoid unnecessary GPU stalls.

---

## Error Handling

Every Vulkan call returning VkResult must be checked.

Never ignore failures.

Unexpected failures should generate descriptive log messages.

---

## Logging

Log:

- Device selection
- Supported Vulkan version
- Enabled features
- Enabled extensions
- Validation layer status

Do not spam logs every frame.

---

## Code Style

Prefer:

- RAII
- std::span
- std::array
- std::vector
- constexpr
- enum class
- strong typing

Avoid:

- Raw owning pointers
- Global mutable state
- Macros unless absolutely necessary

---

## Resource Lifetime

Destroy resources in reverse order of creation.

Wait only when necessary.

Avoid vkDeviceWaitIdle except:

- Application shutdown
- Major resource rebuilds

---

## AI Agent Instructions

When modifying Vulkan code:

1. Preserve validation-clean execution.
2. Prefer modern Vulkan APIs.
3. Do not introduce deprecated synchronization.
4. Do not introduce legacy render passes unless explicitly requested.
5. Keep CPU allocations minimal.
6. Reuse existing abstractions before creating new ones.
7. Explain non-obvious synchronization.
8. Prefer portability over vendor-specific optimizations.
9. Avoid hidden performance regressions.
10. Keep changes localized.

---

## Before Finishing Any Task

Verify that:

- No validation errors are introduced.
- Resources are properly destroyed.
- Synchronization is correct.
- Feature detection is preserved.
- No unnecessary allocations are added.
- The solution follows modern Vulkan best practices.

If multiple implementations are possible, prefer the simplest one that remains performant, maintainable, and compatible across vendors.