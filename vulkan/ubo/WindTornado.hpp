#pragma once

#include <glm/glm.hpp>

// One tornado on the wire: 4 x vec4 = 64 bytes (std140-safe, no padding).
// EVERY component letter is documented here; shaders/ubo/WindField.glsl
// and shaders/includes/vegetation/WindField.glsl must read the same letters.
struct WindTornado {
    glm::vec4 a; // x = baseX (world XZ origin of the funnel), y = baseZ,
                 // z = groundY (world Y of the funnel base; see grounding note
                 //     in WindField.hpp), w = radius (Rankine core radius, metres)
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
