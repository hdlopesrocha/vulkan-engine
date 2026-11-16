#pragma once
// Shared wind-field contract (wind home: VegetationRenderer).
//
// The ambient + tornado wind state lives in ONE uniform buffer on the MAIN
// descriptor set (set=0, binding 27) so every consumer (vegetation vertex,
// fire, SDF, volumetric clouds) samples the identical field. VegetationRenderer
// owns the buffer and packs it; SceneRenderer binds it once at init (static
// set, copied to per-frame sets). Contents stream via host memcpy with
// write-on-change memcmp — never via per-frame descriptor writes.
//
// Grounding note: there is NO runtime CPU height query reachable from the
// updater (HeightFunction::getHeightAt implementations are build-time only:
// PerlinSurface, CachedHeightMapSurface, GradientPerlinSurface; SDF HeightMap
// is GDAL-backed build data). Each tornado therefore carries its own groundY
// (default 0.0 = sea level). GPU heightfield grounding (sampling the live
// terrain height texture in wind_field.glsl) is explicit future work — the
// groundY field is the seam where it plugs in.
#include <glm/glm.hpp>

#include "WindFieldConstants.hpp"
#include "WindTornado.hpp"
#include "WindFieldSettings.hpp"

// Whole shared field: 20 x vec4 = 320 bytes.
struct WindField {
    glm::vec4 ambientA; // x,y,z = reserved (0): the shared field has NO mean
                        //     flow or authored direction — the local wind
                        //     direction is the fBm vector itself. w = fBm/sway
                        //     advection speed (time multiplier)
    // Vector fBm gust field (see windAmbient in
    // shaders/includes/vegetation/WindField.glsl): x = base frequency (1/m),
    // y = octaves (1..kWindFbmMaxOctaves), z = lacunarity (frequency gain per
    // octave), w = gain (amplitude gain per octave, a.k.a. persistence).
    glm::vec4 ambientB;
    // x = amplitude (m/s; peak speed of the 3D fBm gust vector, added to the
    // mean flow), y = vertical weight (0..1 multiplier on the Y component),
    // z = seed (domain offset; different seeds decorrelate the field),
    // w = unused (0).
    glm::vec4 ambientC;
    WindTornado tornadoes[kWindFieldMaxTornadoes];
    glm::vec4 counts;   // x = activeCount (derived at pack time by counting
                        //     active flags — no stale widget counter),
                        // y = wind debug mode (0 off, 1 velocity heat volume,
                        //     2 velocity isosurface, 3 compression heat
                        //     volume, 4 direction volume),
                        // z = isosurface speed in m/s, w = unused (0)
};
static_assert(sizeof(WindField) == 320, "WindField must be 320 bytes");
static_assert(sizeof(WindField) % 16 == 0, "WindField must be a multiple of 16");
static_assert(offsetof(WindField, ambientA) == 0, "WindField.ambientA offset");
static_assert(offsetof(WindField, ambientB) == 16, "WindField.ambientB offset");
static_assert(offsetof(WindField, ambientC) == 32, "WindField.ambientC offset");
static_assert(offsetof(WindField, tornadoes) == 48, "WindField.tornadoes offset");
static_assert(offsetof(WindField, counts) == 304, "WindField.counts offset");
