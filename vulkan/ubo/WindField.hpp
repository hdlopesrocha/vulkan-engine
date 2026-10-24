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
                        // y = wind debug mode (0 off, 1 velocity heat volume,
                        //     2 velocity isosurface, 3 compression heat
                        //     volume, 4 direction volume),
                        // z = isosurface speed in m/s, w = unused (0)
};
static_assert(sizeof(WindField) == 304, "WindField must be 304 bytes");
static_assert(sizeof(WindField) % 16 == 0, "WindField must be a multiple of 16");
static_assert(offsetof(WindField, ambientA) == 0, "WindField.ambientA offset");
static_assert(offsetof(WindField, ambientB) == 16, "WindField.ambientB offset");
static_assert(offsetof(WindField, tornadoes) == 32, "WindField.tornadoes offset");
static_assert(offsetof(WindField, counts) == 288, "WindField.counts offset");
