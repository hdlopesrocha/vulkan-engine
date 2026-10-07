#pragma once
// Shared wind-field contract (wind home: VegetationRenderer).
//
// The ambient + tornado wind state lives in ONE uniform buffer on the MAIN
// descriptor set (set=0, binding 27) so every consumer (vegetation vertex,
// fire, SDF) samples the identical field. VegetationRenderer owns the buffer
// and packs it; SceneRenderer binds it once at init (static set, copied to
// per-frame sets). Contents stream via host memcpy with write-on-change
// memcmp — never via per-frame descriptor writes.
//
// Grounding note: there is NO runtime CPU height query reachable from the
// updater (HeightFunction::getHeightAt implementations are build-time only:
// PerlinSurface, CachedHeightMapSurface, GradientPerlinSurface; SDF HeightMap
// is GDAL-backed build data). Each tornado therefore carries its own groundY
// (default 0.0 = sea level). GPU heightfield grounding (sampling the live
// terrain height texture in wind_field.glsl) is explicit future work — the
// groundY field is the seam where it plugs in.
#include <glm/glm.hpp>
#include <array>
#include <cstdint>

inline constexpr uint32_t kWindFieldMaxTornadoes = 4;
inline constexpr uint32_t kWindFieldBinding = 27;

// One tornado on the wire: 4 x vec4 = 64 bytes (std140-safe, no padding).
// EVERY component letter is documented here; shaders/ubo/WindField.glsl
// and shaders/includes/vegetation/WindField.glsl must read the same letters.
struct WindTornado {
    glm::vec4 a; // x = baseX (world XZ origin of the funnel), y = baseZ,
                 // z = groundY (world Y of the funnel base; see grounding note
                 //     above), w = radius (Rankine core radius, metres)
    glm::vec4 b; // x = height (funnel height, metres; updraft decays to zero
                 //     at groundY + height), y = strength (max tangential
                 //     speed in m/s at the core edge), z = direction sign
                 //     (-1 = clockwise, +1 = counter-clockwise seen from
                 //     above, 0 = pure updraft column), w = phase (seconds;
                 //     staggers the wander orbit and direction swing)
    glm::vec4 c; // x = delta_time (per-tornado time scale multiplier on the
                 //     consumer's clock; 1 = real time, 0 = frozen),
                 // y = active flag (1 = active, 0 = skipped by windSVF),
                 // z = wander radius (metres; centre orbits this far from the
                 //     drifted base), w = wander speed (radians/second of the
                 //     centre orbit; negative orbits the other way)
    glm::vec4 d; // x = driftVelX (m/s; constant base translation),
                 // y = driftVelZ (m/s),
                 // z = swing amplitude (0 = static direction; otherwise 0..1
                 //     blend from the base sign toward a cos oscillation),
                 // w = swing frequency (radians/second of the swing)
};
static_assert(sizeof(WindTornado) == 64, "WindTornado must be 64 bytes");
static_assert(sizeof(WindTornado) % 16 == 0, "WindTornado must be a multiple of 16");
static_assert(offsetof(WindTornado, a) == 0, "WindTornado.a offset");
static_assert(offsetof(WindTornado, b) == 16, "WindTornado.b offset");
static_assert(offsetof(WindTornado, c) == 32, "WindTornado.c offset");
static_assert(offsetof(WindTornado, d) == 48, "WindTornado.d offset");

// Whole shared field: 19 x vec4 = 304 bytes.
struct WindField {
    glm::vec4 ambientA; // x = windDirX, y = windDirZ (normalized at pack time;
                        //     mirrored from VegetationRenderer::WindSettings,
                        //     never duplicated as sliders), z = strength
                        //     (m/s), w = advection speed (gust time multiplier)
    glm::vec4 ambientB; // x = baseFrequency, y = gustFrequency,
                        // z = gustStrength (0 = laminar), w = unused (0)
    WindTornado tornadoes[kWindFieldMaxTornadoes];
    glm::vec4 counts;   // x = activeCount (derived at pack time by counting
                        //     active flags — no stale widget counter),
                        // y = wind debug mode (0 off, 1 heat map,
                        //     2 velocity isosurface),
                        // z = isosurface speed in m/s, w = unused (0)
};
static_assert(sizeof(WindField) == 304, "WindField must be 304 bytes");
static_assert(sizeof(WindField) % 16 == 0, "WindField must be a multiple of 16");
static_assert(offsetof(WindField, ambientA) == 0, "WindField.ambientA offset");
static_assert(offsetof(WindField, ambientB) == 16, "WindField.ambientB offset");
static_assert(offsetof(WindField, tornadoes) == 32, "WindField.tornadoes offset");
static_assert(offsetof(WindField, counts) == 288, "WindField.counts offset");

// Widget-facing settings (edited by the wind widget; packed to the wire
// struct by VegetationRenderer::updateWindFieldUBO). Ambient sliders are NOT
// duplicated here — ambient is mirrored from WindSettings at pack time.
struct TornadoSettings {
    glm::vec2 baseXZ = glm::vec2(0.0f, 0.0f);
    float groundY = 0.0f; // sea level default; see grounding note above
    float radius = 25.0f;
    float height = 120.0f;
    float strength = 30.0f;
    float direction = 1.0f; // -1..1
    float phase = 0.0f;
    float deltaTime = 1.0f;
    glm::vec2 driftVelocity = glm::vec2(0.0f, 0.0f);
    float wanderRadius = 0.0f;
    float wanderSpeed = 0.0f;
    float swingAmplitude = 0.0f; // 0 = static direction
    float swingFrequency = 0.0f;
    bool active = false;
};

struct WindFieldSettings {
    std::array<TornadoSettings, kWindFieldMaxTornadoes> tornadoes{};
    // Wind debug mode shown by the composite overlay: 0 = off, 1 = heat +
    // direction raymarch, 2 = velocity isosurface. Owned by the ray marching
    // widget (not the wind widget: it visualizes, it doesn't simulate).
    int windDebugMode = 0;
    // Isosurface speed for mode 2 (m/s): the surface where |wind| = iso.
    float windDebugIso = 8.0f;
};
