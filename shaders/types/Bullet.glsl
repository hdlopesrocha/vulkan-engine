#ifndef BULLET_GLSL
#define BULLET_GLSL

// Canonical bullet interaction state (std430, 48 B). Single definition shared
// with the CPU: sdf/types/Bullet.hpp — identical layout, names and
// offsets.
// Offset table (verified against the C++ static_asserts):
//   start 0, radiusStart 12, velocity 16, pathLength 28, radiusEnd 32,
//   loopDuration 36, intensity 40, phase 44.
// start/velocity are stored in the smoke shape's LOCAL frame (the generic
// SdfModel owns the transform; the shader never transforms a bullet).
struct Bullet {
    vec3 start;        // offset  0  path start (smoke-local)
    float radiusStart; // offset 12  radius at launch (m)
    vec3 velocity;     // offset 16  velocity (m/s, smoke-local; magnitude = speed)
    float pathLength;  // offset 28  carved path length (m)
    float radiusEnd;   // offset 32  radius at the head (m)
    float loopDuration;// offset 36  per-bullet cycle (s; 0 = one-shot)
    float intensity;   // offset 40  0 = empty slot
    float phase;       // offset 44  launch time within the loop (s)
};

#endif // BULLET_GLSL
