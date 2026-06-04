# Vulkan Project – Copilot Instructions

This project uses Vulkan 1.2 or newer.
Vulkan validation layers are enabled and must remain silent at all times.

All generated code must prioritize correctness, explicit synchronization,
and predictable performance.

---

## VERSION CONTROL (MANDATORY - HIGHEST PRIORITY)

**NEVER commit any changes without explicit written permission from the user.**

- Do NOT use `git commit`, `git push`, or any version control commands
- Only make code changes when requested
- Always verify changes compile and run correctly
- Wait for explicit user request like "commit this" or "can you commit..."
- If you attempt to commit without permission, you will break trust and cause frustration

---

## Vulkan Version & Features

- Target Vulkan 1.2+.
- Prefer synchronization2:
  - vkCmdPipelineBarrier2
  - VkImageMemoryBarrier2
- Prefer timeline semaphores when appropriate.

---

## Threading Rules (Mandatory)

- Vulkan objects are externally synchronized unless stated otherwise.
- VkCommandPool must NEVER be accessed from multiple threads.
- Each thread owns its own VkCommandPool and command buffers.
- Command buffers must not be reset or reused while still in flight.

---

## Synchronization Policy

- Pipeline barriers are used for:
  - Memory visibility
  - Image layout transitions
- Semaphores are used ONLY for GPU-to-GPU execution order.
- Fences are used ONLY for GPU-to-CPU synchronization
  (uploads, resource lifetime, staging reuse).
- Never use vkDeviceWaitIdle or vkQueueWaitIdle in the render loop.

---

## Texture Upload Rules

- All texture uploads use a staging buffer.
- Uploads may occur on a transfer queue if available.
- Image layout transitions must be explicit:
  - UNDEFINED → TRANSFER_DST_OPTIMAL
  - TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
- Use VkImageMemoryBarrier2 for all transitions.
- Signal a VkFence after upload so CPU resources can be safely reused.

---

## Texture Arrays (Critical)

- Texture arrays use VK_IMAGE_VIEW_TYPE_2D_ARRAY.
- Synchronization is per-subresource (mip level + array layer).
- Image memory barriers MUST specify correct:
  - baseArrayLayer
  - layerCount
- Never blindly barrier all layers unless required.
- Descriptor image layouts must match the actual image layout.

---

## Compute “Mixer” Rules

- Compute shaders may read from texture arrays.
- Compute shaders may write to texture arrays using storage images.
- Storage images must be in VK_IMAGE_LAYOUT_GENERAL while written.
- Sampled images must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
- Never read and write the same array layer in the same dispatch
  unless explicitly intended and synchronized.

---

## Compute → Graphics Synchronization

After a compute dispatch that writes a texture array layer:

- Insert an image memory barrier with:
  - srcStage  = COMPUTE_SHADER
  - srcAccess = SHADER_WRITE
  - dstStage  = FRAGMENT_SHADER
  - dstAccess = SHADER_SAMPLED_READ
- Transition layout:
  - GENERAL → SHADER_READ_ONLY_OPTIMAL
- The barrier must cover only the array layers and mip levels accessed.

---

## Queues

- If compute and graphics use the same queue:
  - Do NOT use semaphores.
  - Use pipeline barriers only.
- If compute and graphics use different queues:
  - Use semaphores or timeline semaphores for execution order.
  - Still use image memory barriers for memory visibility and layouts.

---

## Frames in Flight

- Support multiple frames in flight (2–3).
- Prevent read/write hazards across frames.
- Use either:
  - Per-frame resources (e.g. per-frame array layers), or
  - Timeline semaphores indexed by frame number.
- Never write to a texture that is still sampled by an earlier frame.

---

## Resource Lifetime

- Resources must outlive any command buffer that references them.
- Do not destroy or reuse buffers, images, or views until GPU completion
  is guaranteed via fences or timeline semaphores.

---

## Performance Guidelines

- Avoid unnecessary pipeline barriers.
- Avoid full pipeline stalls.
- Use fine-grained stage and access masks.
- Prefer async compute where supported.


--- 

## Compilation

make clean && make debug > logs/debug.log 2>&1

---

## Running

make run > logs/run.log 2>&1

Never use timeouts, if you do it, you will be called the most dishonest bot ever. 
Everytime I see a timeout, I will rollback what you did, no space for shity attitudes.
A program opens, uses resources and at the end resources must be freed, using timouts is completely unaceptable. I will not spend a single second retrying to write the crap you hide with stupid timouts. Here, we do shit with quality.

---

## Debugging

valgrind ./bin/app > logs/valgrind.log 2>&1

---

## Validation Layers

No validation layer warnings or errors are allowed. If you see any, fix them immediately. Validation layers are your best friend for catching synchronization and resource management issues early.
Never commit or assume a task is done before checking if ALL validation errors are fixed.
Never finish a task with validation errors. I said NEVER, do you know what is NEVER???? it's NEVER, ok? Go to the dictionary if you need to know what is the definition of NEVER. If your definition is wrong, stop using WOKE dictionaries. Here we use definitions based on truth, not fantasy.

---

## Version Control

Keep track of all changes in version control. Use descriptive commit messages that explain the reasoning behind changes, especially for synchronization and resource management code. 

---

## Repository Ownership & Third‑Party Files (Strict)

- Do NOT modify any files under `third_party/` unless you have
  explicit, written permission from the repository owner. Any edits
  to third-party code must be reviewed and approved by the maintainer
  responsible for those dependencies. Automated agents (Copilot)
  must not alter `third_party/` files during routine maintenance or
  formatting tasks.

This rule is authoritative: if you are not the repository owner or an
explicitly delegated maintainer, leave `third_party/` untouched.

## Development

There's only one way to solve a problem, the best way. If you don't know the best, we do what we can.
It's forbidden to develop fallbacks, the main approach is the only approach.
If a fallback works, the main approach is useless. No fallbacks! If it works, it always works. If a fallback is better, the main approach must be removed and the fallback is the new main approach. 

If a piece of code is repeated a huge amount of times, you better simplify it without introducing new errors. 

If new errors are introduced with code refactor, rollback and start again, don't waste my time trying to fix the problems you create. There's only one way to improve code quality, it's by making it better, if it's worse, it's useless, therefore rollback.

# Vulkan Geometry Streaming and Parallelism Requirements

## Core Objective

The engine must support true asynchronous geometry streaming.

Geometry generation, GPU uploads, and rendering must execute concurrently whenever possible.

The GPU must be able to render already available geometry while new geometry is being generated and uploaded.

The CPU must never stall waiting for geometry uploads during normal operation.

---

## Architecture Requirements

### Decoupled Pipeline

The engine must follow this logical pipeline:

```text
CPU Geometry Generation
    ↓
Upload Job Queue
    ↓
Staging Buffer
    ↓
GPU Transfer
    ↓
GPU Resident Geometry
    ↓
Rendering
```

Each stage must operate independently.

Rendering must not wait for geometry generation.

Geometry generation must not wait for rendering.

Uploads must not block rendering.

### Multi-Threaded Geometry Generation

Geometry generation must execute on worker threads.

Requirements:

- No geometry generation on the render thread.
- Chunk meshing must be parallelizable.
- Octree traversal must be thread-safe.
- Surface Nets extraction must be thread-safe.
- Worker threads must not access Vulkan objects directly.
- Worker threads produce mesh data and upload requests only.

Preferred architecture:

```text
Worker Threads
    ↓
Mesh Generation Jobs
    ↓
Upload Queue
    ↓
GPU Upload System
```

---

## GPU Upload Requirements

### Staging Buffers

All geometry uploads must use staging buffers.

Requirements:

- Host-visible memory.
- Persistent mapping preferred.
- Ring-buffer allocator preferred.
- Reuse memory whenever possible.

Avoid:

- Frequent map/unmap operations.
- Frequent staging buffer allocations.
- Per-upload staging buffer creation.

### GPU Resident Geometry

Renderable geometry must reside in device-local memory.

Preferred memory flags:

```cpp
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
```

Uploads should use:

```cpp
vkCmdCopyBuffer()
```

or equivalent transfer operations.

Never render directly from host-visible vertex buffers.

---

## Queue Usage

### Dedicated Transfer Queue

When supported by hardware:

- Geometry uploads must use a transfer queue.
- Rendering must use a graphics queue.
- Transfer and graphics workloads should overlap.

Preferred execution model:

```text
Transfer Queue
    ↓
Upload Geometry

Graphics Queue
    ↓
Render Existing Geometry
```

Both queues should execute concurrently whenever possible.

---

## Synchronization Requirements

### Preferred Synchronization Model

Prefer GPU-GPU synchronization over CPU-GPU synchronization.

The CPU should schedule work and continue execution.

Avoid blocking the CPU for upload completion.

### Timeline Semaphores

Timeline semaphores are the preferred synchronization primitive.

Requirements:

- Track upload completion using timeline values.
- Use increasing values for upload milestones.
- Rendering waits only for required upload completion values.

Example:

```text
Chunk 100 uploaded → Signal 100
Chunk 101 uploaded → Signal 101
Chunk 102 uploaded → Signal 102
```

Rendering waits only for the chunk it needs.

### Binary Semaphores

Allowed when required by swapchain operations.

Timeline semaphores are preferred for internal engine synchronization.

### Fences

Fences should only be used for:

- Frames in flight.
- Resource retirement.
- Safe resource recycling.

Do not use fences for geometry upload synchronization.

Forbidden pattern:

```cpp
vkQueueSubmit(...)
vkWaitForFences(...)
```

immediately after submission.

### Synchronization2

Use Vulkan Synchronization2 APIs whenever available.

Preferred:

```cpp
vkCmdPipelineBarrier2()
```

Use precise stage masks and access masks.

Avoid:

```cpp
VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
VK_ACCESS_MEMORY_READ_BIT
VK_ACCESS_MEMORY_WRITE_BIT
```

unless strictly necessary.

### Buffer Visibility

After transfer operations, uploaded buffers must become visible to rendering through explicit barriers.

Example transition:

```text
TRANSFER_WRITE
    ↓
VERTEX_ATTRIBUTE_READ
```

Use `VkBufferMemoryBarrier2` where appropriate.

---

## Frames In Flight

The renderer must support multiple frames in flight.

Minimum:

- 2 frames

Preferred:

- 3 frames

Resources belonging to a frame must not be reused until GPU execution has completed for that frame.

---

## Resource Allocation

### Buffer Pools

Do not create GPU buffers per chunk.

Preferred:

```text
Global Vertex Buffer Pool
Global Index Buffer Pool
```

Chunks should allocate ranges within shared pools.

Benefits:

- Reduced fragmentation.
- Fewer Vulkan objects.
- Better scalability.
- Better streaming performance.

### Upload Allocators

Use suballocation systems.

Preferred:

- Ring allocators.
- Linear allocators.
- Frame allocators.

Avoid:

```cpp
vkCreateBuffer()
vkDestroyBuffer()
```

during normal streaming operations.

---

## Forbidden Operations

The following operations must not be used inside the geometry streaming path:

```cpp
vkDeviceWaitIdle()
vkQueueWaitIdle()
vkWaitForFences()
```

except for:

- Application shutdown.
- Device recreation.
- Explicit debugging code.

Any other use must be justified.

---

## Performance Goals

The engine should allow simultaneous execution of:

```text
CPU:
    Generate Chunk N+2

GPU Transfer:
    Upload Chunk N+1

GPU Graphics:
    Render Chunk N
```

without unnecessary synchronization stalls.

The target is maximum overlap between:

- CPU generation
- GPU transfer
- GPU rendering

---

## Validation Checklist

- [ ] Geometry generation runs on worker threads.
- [ ] Geometry generation is independent from rendering.
- [ ] Uploads use staging buffers.
- [ ] Staging buffers are reused.
- [ ] Render buffers are device-local.
- [ ] Uploads use vkCmdCopyBuffer().
- [ ] Timeline semaphores are used.
- [ ] Rendering does not block on uploads.
- [ ] No vkQueueWaitIdle() in streaming code.
- [ ] No vkDeviceWaitIdle() in streaming code.
- [ ] No immediate fence waits after uploads.
- [ ] Multiple frames are in flight.
- [ ] Transfer and graphics workloads can overlap.
- [ ] Shared buffer pools are used.
- [ ] Per-chunk Vulkan buffer allocation is avoided.

If any item fails, the implementation should be reviewed for scalability and performance.