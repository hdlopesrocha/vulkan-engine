#ifndef WIND_FIELD_BLOCK_GLSL
#define WIND_FIELD_BLOCK_GLSL

// Exact GLSL mirror of vulkan/ubo/WindField.hpp (std140 uniform block
// member). Offsets must match the C++ side 1:1:
//   ambientA @ 0, ambientB @ 16, tornadoes[i] @ 32 + i*64, counts @ 288,
//   total size 304 bytes. EVERY component letter is documented here; keep in
//   sync with WindField.hpp.

struct WindTornado {
    vec4 a; // x = baseX (world XZ origin), y = baseZ, z = groundY (world Y of
            //     funnel base), w = radius (Rankine core radius, m)
    vec4 b; // x = height (funnel height, m), y = strength (max tangential m/s
            //     at core edge), z = direction sign (-1 = CW, +1 = CCW from
            //     above, 0 = pure updraft), w = phase (s)
    vec4 c; // x = delta_time (per-tornado time scale), y = active flag
            //     (1 = active), z = wander radius (m), w = wander speed (rad/s)
    vec4 d; // x = driftVelX (m/s), y = driftVelZ (m/s), z = swing amplitude
            //     (0 = static direction), w = swing frequency (rad/s)
};

struct WindField {
    vec4 ambientA; // xy = wind dir XZ (normalized), z = strength (m/s),
                   // w = advection speed (gust time multiplier)
    vec4 ambientB; // x = baseFrequency, y = gustFrequency, z = gustStrength,
                   // w = unused
    WindTornado tornadoes[4]; // WIND_FIELD_MAX_TORNADOES entries, 64 bytes each
    vec4 counts; // x = activeCount, y = wind debug mode (0 off, 1 heat map,
                   //     2 velocity isosurface), z = isosurface speed (m/s), w = unused
};

#define WIND_FIELD_MAX_TORNADOES 4

#endif // WIND_FIELD_BLOCK_GLSL
