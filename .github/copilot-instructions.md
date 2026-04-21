# Vulkan Project – Copilot Instructions

This project uses Vulkan 1.2 or newer.
Vulkan validation layers are enabled and must remain silent at all times.

All generated code must prioritize correctness, explicit synchronization,
and predictable performance.

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

make clean && make debug > /tmp/xxx.log 2>&1

---

## Running

make run > /tmp/xxx.log 2>&1

---

## Debugging

valgrind ./bin/app > /tmp/xxx.log 2>&1

---

## Validation Layers

No validation layer warnings or errors are allowed. If you see any, fix them immediately. Validation layers are your best friend for catching synchronization and resource management issues early.
Never commit or assume a task is done before checking if ALL validation errors are fixed.

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