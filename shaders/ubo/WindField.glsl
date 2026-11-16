#ifndef WIND_FIELD_BLOCK_GLSL
#define WIND_FIELD_BLOCK_GLSL

// Exact GLSL mirror of vulkan/ubo/WindField.hpp (std140 uniform block
// member). Offsets must match the C++ side 1:1:
//   ambientA @ 0, ambientB @ 16, ambientC @ 32, tornadoes[i] @ 48 + i*64,
//   counts @ 304, total size 320 bytes. EVERY component letter is documented
//   here; keep in sync with WindField.hpp.

struct WindTornado {
    vec4 a; // x = baseX (world XZ origin), y = baseZ, z = groundY (world Y of
            //     funnel base), w = radius (Rankine core radius, m)
    vec4 b; // x = height (funnel height, m), y = strength (max tangential m/s
            //     at core edge), z = direction sign (-1 = CW, +1 = CCW from
            //     above, 0 = pure updraft), w = phase (s)
    vec4 c; // x = delta_time (per-tornado time scale), y = active flag
            //     (1 = active), z = wander radius (m), w = wander speed (rad/s)
    vec4 d; // x = driftVelX (m/s), y = driftVelZ (m/s), z = swing amplitude
            // (0 = static direction), w = swing frequency (rad/s)
};

struct WindField {
    vec4 ambientA; // xyz reserved (0): no mean flow/authored direction — the
                   //      local direction is the fBm vector itself;
                   //      w = fBm/sway advection speed (time multiplier)
    // Vector fBm gust field (windAmbient): x = base frequency (1/m),
    // y = octaves (1..WIND_FBM_MAX_OCTAVES), z = lacunarity (frequency gain
    // per octave), w = gain (amplitude gain per octave, a.k.a. persistence).
    vec4 ambientB;
    // x = amplitude (m/s; peak speed of the 3D fBm gust vector added to the
    // mean flow), y = vertical weight (0..1 on the Y component), z = seed
    // (domain offset), w = unused (0).
    vec4 ambientC;
    WindTornado tornadoes[4]; // WIND_FIELD_MAX_TORNADOES entries, 64 bytes each
    vec4 counts; // x = activeCount, y = wind debug mode (0 off,
                 //     1 velocity heat volume, 2 velocity isosurface,
                 //     3 compression heat volume, 4 direction volume),
                 //     z = isosurface speed (m/s), w = unused
};

#define WIND_FIELD_MAX_TORNADOES 4
// Must match vulkan/ubo/WindFieldConstants.hpp kWindFbmMaxOctaves (GLSL needs
// a compile-time loop bound; the active count comes from ambientB.y).
#define WIND_FBM_MAX_OCTAVES 6

#endif // WIND_FIELD_BLOCK_GLSL
