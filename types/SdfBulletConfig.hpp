#pragma once

// Bullet defaults stamped into SSBO slots on fire (slot 0 = auto-loop
// template rebuilt live from these every loop).
struct SdfBulletConfig {
    float radius = 16.0f;      // tunnel radius (m)
    float speed = 2.0f;       // flight speed (m/s)
    float length = 512.0f;     // path length (m)
    float angleDeg = 0.0f;     // XZ plane, 0 = +X
    bool autoFire = true;      // re-fire from defaults every smoke loop
};
