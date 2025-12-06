## Quick orientation

This is a minimal Vulkan starter project contained in a single source file `main.cpp` with a small `Makefile` that builds an executable called `app`.

- Entry point: `main.cpp` — defines `VulkanApp` and `main()` which constructs and runs the app.
- Build: top-level `Makefile` (target `all`) uses `pkg-config` for `glfw3` and `vulkan` and outputs `./app`.

## Big-picture architecture (what to know fast)

- Single-class structure: `VulkanApp` encapsulates lifecycle: `initWindow()`, `initVulkan()`, `mainLoop()`, `cleanup()`.
- Core Vulkan flow present and discoverable in `main.cpp`: create VkInstance -> setup debug messenger -> create surface -> pick physical device -> create logical device.
- Important helper functions to reference when making changes: `createInstance`, `setupDebugMessenger`, `createSurface`, `pickPhysicalDevice`, `createLogicalDevice`, `findQueueFamilies`, `checkValidationLayerSupport`.
- Data flow: `VkInstance` -> `VkPhysicalDevice` -> `VkDevice` and queue families; `surface` is created from GLFW window and used for present support checks.

## Build / run / debug (developer workflows)

- Build (linux): run `make`. This compiles `main.cpp` into `./app` using `g++` with `pkg-config --cflags/--libs glfw3 vulkan`.
- Clean: `make clean` removes `./app`.
- Validation layers: the binary enables validation layers by default unless `NDEBUG` is defined. That is controlled in `main.cpp`:
  - If `NDEBUG` is defined -> validation disabled
  - Otherwise -> validation enabled
  To run with validation layers disabled, compile with `-DNDEBUG` added to `CFLAGS`.
- System dependencies: `libglfw3` and Vulkan SDK/system driver are required. The `Makefile` also exposes an `install` target that runs `sudo apt install vulkan-validationlayers`, but you may also need GLFW dev packages and the Vulkan SDK (distribution dependent).

## Project-specific conventions & patterns

- Small, single-file prototype: most changes will happen in `main.cpp`. Keep method-level helpers as private methods of `VulkanApp` to match the existing style.
- Naming: camelCase for methods (`createInstance`, `pickPhysicalDevice`) and `PascalCase` for types/structs inside the file (`QueueFamilyIndices`).
- Error handling: functions throw `std::runtime_error` on fatal errors and the `main()` prints exceptions to `stderr` and returns non-zero.
- Debug: debug callback `debugCallback` prints validation messages to `stderr`. The debug messenger is created with `VK_EXT_debug_utils` when layers are enabled.

## Integration points & external dependencies

- GLFW windowing: `glfwCreateWindow` + `glfwCreateWindowSurface(instance, ...)` — any change to windowing must preserve these calls.
- Vulkan loader and layers: uses `vkCreateInstance` and optional `vkCreateDebugUtilsMessengerEXT` via `vkGetInstanceProcAddr`.
- Packages used via `pkg-config`: `glfw3` and `vulkan` (Makefile relies on system pkg-config entries).

## Examples of concrete AI prompts (use these to implement features safely)

- "Add basic swapchain creation to this project: create a `createSwapchain()` private method in `VulkanApp`, call it from `initVulkan()` after `createSurface()`, and introduce member variables `VkSwapchainKHR swapchain` and `std::vector<VkImage> swapchainImages`. Keep existing error handling style and naming conventions."

- "Refactor `VulkanApp` into `src/engine.cpp` and `include/engine.h`: extract private methods as `protected`/`private` members and update `Makefile` to compile both files. Keep existing compile flags and pkg-config usage."

- "Add a small unit-style smoke test that runs the app headless (no window) to verify `createInstance()` and `pickPhysicalDevice()` succeed — keep runtime checks and print minimal pass/fail messages to stdout."

## Files to look at when changing behavior

- `main.cpp` — the single source of truth for current behavior.
- `Makefile` — compile flags, pkg-config usage, `install` target.

## Known limitations & safe assumptions

- This is a minimal starter. There is no swapchain, render pass, or command buffer code yet. Expect to add those pieces for any rendering changes.
- The project assumes a Linux-like environment with `pkg-config` entries for `glfw3` and `vulkan`.

If any section is unclear or you'd like more detail (for example, a suggested file split, recommended tests, or a hardware-independent CI configuration), tell me which part to expand and I will iterate.
