# Vulkan Engine Project

## Overview

This project is a minimal yet extensible 3D engine prototype built with Vulkan and C++. It provides a foundation for real-time graphics applications, leveraging the Vulkan API for high-performance rendering and modern GPU features. The engine is structured for rapid prototyping and experimentation, with a focus on clarity and modularity.

#### SDF & Surface Nets Mesh Generation
The engine uses Signed Distance Functions (SDF) to procedurally generate geometry. It implements the Surface Nets algorithm to extract and mesh isosurfaces from SDF volumes, enabling efficient and flexible creation of complex 3D shapes and terrains.

### Vulkan Functionality
- Uses the Vulkan API for explicit, low-level graphics rendering.
- Handles instance, device, and surface creation, with validation layer support for debugging.
- Designed for easy extension to swapchain, render pass, and command buffer management.
- Integrates with GLFW for cross-platform windowing.

### Octree and Spatial Data Structures
- Includes octree and bounding volume (box, sphere, cube) classes for efficient spatial partitioning and scene management.
- Supports fast queries for collision detection, frustum culling, and hierarchical scene traversal.
- Math and geometry utilities are provided in the `math/` directory, including bounding boxes, spheres, and transformation helpers.

## Build and Run Instructions

### Prerequisites
- Linux system with Vulkan SDK and drivers installed
- `libglfw3` and `libglfw3-dev` packages
- C++ compiler (e.g., `g++`)
- `pkg-config` utility

### Build
To compile the project, run:

```sh
make 
```

This will build the main application and output the executable to `bin/app`.

### Clean
To remove the built executable and object files, run:

```sh
make clean
```

### Run
To launch the application after building:

```sh
make run
```


### Debug Build
To build with debug symbols and validation layers enabled:

```sh
make debug
```

## Directory Structure
- `main.cpp` — Entry point and Vulkan setup
- `math/` — Math, geometry, and spatial partitioning (octree, bounding volumes)
- `events/` — Event system for input and window management
- `bin/` — Compiled binaries and runtime files
- `shaders/`, `textures/` — Graphics assets

## License
See LICENSE for details.
# vulkan-engine
Physics Engine in Vulkan
