#ifndef SMOKE_BULLET_STATE_GLSL
#define SMOKE_BULLET_STATE_GLSL

// Extracted from shaders/includes/sdf_smoke.glsl (single-struct GLSL type).

// Per-bullet runtime state. Motion is a pure GPU function of global time:
// every bullet loops with period c.y, offset by phase c.w, so several
// bullets can be staggered and the auto bullet can wait for the smoke bloom.
struct SmokeBulletState {
    bool live;        // intensity > 0 and within the pass + refill grace
    bool headOnPath;  // head has not left the carved path yet
    float traveled;   // head distance along path (m)
    float age;        // seconds since this loop's pass started
};

#endif // SMOKE_BULLET_STATE_GLSL
