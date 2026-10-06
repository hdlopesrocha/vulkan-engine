#ifndef SDF_CONTAINER_GLSL
#define SDF_CONTAINER_GLSL

// Canonical SDF container (std430, 48 B). Single definition shared with the
// CPU scene builder: sdf/types/SdfContainer.hpp — identical layout, names
// and offsets.
// Offset table (verified against the C++ static_asserts):
//   boundsMin 0, boundsMax 16, resX 28, resY 32, resZ 36, cellStart 40.
// boundsMax starts at 16 (std430 vec3 alignment); 44..48 is std140-style size rounding to 16.
// Per-container instance membership is CPU-only and not part of this struct.
struct SdfContainer {
    vec3 boundsMin; // offset  0  world AABB min
    vec3 boundsMax; // offset 16  world AABB max
    uint resX;      // offset 28  grid resolution X (> 0)
    uint resY;      // offset 32
    uint resZ;      // offset 36
    uint cellStart; // offset 40  global cell-buffer start
};

#endif // SDF_CONTAINER_GLSL
