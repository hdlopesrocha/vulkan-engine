#pragma once

// Primitive type encoded in SdfDefinition::meta.x.
// GLSL twin: shaders/types/SdfPrimitiveType.glsl (SDF_PRIM_*).
#include <cstdint>

enum class SdfPrimitiveType : uint32_t {
    Sphere = 0,
    Box = 1,
    RoundedBox = 2,
    Capsule = 3,
    Cylinder = 4,
    Cone = 5,
    Torus = 6,
    Plane = 7,
    // Tapered, base-anchored flame: exact round cone (sphere-capped) from
    // (0,0,rBase) to (0,height,rTip) in local space, i.e. a rounded capsule
    // with two different radii. Base sits at the instance point, axis +Y
    // (aligned to the surface normal via instance euler). Spikes are added
    // procedurally in-shader (see params1).
    Flame = 8,
    // Smoke bomb volume: sphere domain whose radius grows on a looped time
    // curve (see params0); layered-noise density, wind, bullet interaction
    // and lighting are evaluated procedurally in-shader from the smoke
    // state buffer (set=1 binding 8). Marching uses the grown radius;
    // density handles growth envelope, tunnel, wake and refill.
    Smoke = 9,
    // Rock boulder: Perlin-displaced sphere (static). params0.x = base
    // radius, .y = noise frequency (per local unit), .z = displacement
    // amplitude as a fraction of the radius; the instance seed offsets the
    // noise lattice so every rock is unique. Bounds add the displacement.
    Rock = 10
};
