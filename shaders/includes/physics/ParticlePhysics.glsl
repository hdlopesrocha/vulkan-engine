// Generic GPU particle physics: canonical particle record, spatial-hash broad
// phase, SDF/plane narrow phase and a Jacobi-style impulse solver.
//
// Everything here is state-agnostic except the particle record layout, which
// is the canonical one: { position, radius, velocity, mass, prevPosition,
// flags } at set 0 binding 0. A consumer includes this file, keeps that
// layout and drives the passes from its own dispatch (see
// shaders/computation/physics/ParticlePhysics.comp). Rendering is completely
// independent: a consumer may draw its particles any way it likes (the water
// droplets ray-march them) and the collision geometry never changes with it.
#include "WorldSdf.glsl"

struct Particle {
    vec3 position;
    float radius;
    vec3 velocity;
    float mass;
    vec3 prevPosition;
    uint flags;
};

layout(std430, set = 0, binding = 0, scalar) buffer ParticleState { Particle particles[]; };

// particle slots (a bucket that overflows drops its excess particles).
layout(std430, set = 0, binding = 1, scalar) buffer GridCounts  { uint gridCounts[];  };
layout(std430, set = 0, binding = 2, scalar) buffer GridEntries { uint gridEntries[]; };
// [0] = (overflowCount, pairContacts, terrainContacts, maxContactsPerParticle).
// Pass 4 adds its terrain contacts to .z; pass 5 has no dedicated slot in the
// frozen interface and counts particles that left the region there as well, so
// the host reads .z as "terrain contacts + escaped particles".
layout(std430, set = 0, binding = 3, scalar) buffer GridExtra   { uvec4 gridExtra[]; };

// Push constants shared by every pass of a particle simulation.
layout(push_constant) uniform PC {
    vec4 spawnCenterRadius;  // xyz = spawn sphere center, w = spawn sphere radius
    vec4 regionMin;          // xyz = simulation region min, w = water level (y)
    vec4 regionMax;          // xyz = simulation region max, w = grid cell size
    vec4 physics;            // x = gravity (positive, downward), y = restitution, z = friction, w = damping (per second)
    vec4 misc;               // x = dt, y = particleCount, z = resetSeed (uint bits), w = pass
    vec4 spawnParams;        // x = spawn mode (0 ball, 1 box), y = spawn speed (m/s), zw reserved
} pc;

const uint GRID_BUCKETS = 8192u;
const uint GRID_MAX_PER_BUCKET = 64u;
const uint LOCAL_SIZE = 64u;

const uint FLAG_ACTIVE = 1u;

// The solver works in velocity units (unit-mass contact), so the tangential
// impulse is the tangential speed change clamped to the Coulomb cone
// friction * |jn| in the same units.
const float FRICTION_FACTOR = 1.0;

// FINALIZE tuning: below SETTLE_SPEED the droplet is bled of its residual
// velocity at SETTLE_DAMPING per second (strong, but frame-rate independent).
const float SETTLE_SPEED = 0.02;
const float SETTLE_DAMPING = 10.0;


// xorshift32 (same stream as VegetationPlacement.comp).
uint xorshift32(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(inout uint state) {
    return float(xorshift32(state) & 0x00FFFFFFu) / float(0x01000000u);
}

// The multiplies run in uint so negative cell coordinates wrap deterministically
// instead of hitting signed-integer-overflow undefined behaviour.
uint cellHash(ivec3 c) {
    return (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u);
}


// Particle count carried in the float push-constant slot (misc.y).
uint simParticleCount() {
    return uint(max(0.0, pc.misc.y));
}

bool isActive(uint i) {
    return (particles[i].flags & FLAG_ACTIVE) != 0u;
}

bool insideRegion(vec3 pos) {
    return all(greaterThanEqual(pos, pc.regionMin.xyz))
        && all(lessThanEqual(pos, pc.regionMax.xyz));
}

// INTEGRATE: semi-implicit Euler. Gravity is positive-downward in the push
// constants, damping is a per-second factor applied multiplicatively.
void integrateParticle(uint i) {
    if (!isActive(i)) {
        return;
    }
    float dt = pc.misc.x;
    vec3 v = particles[i].velocity;
    v.y -= pc.physics.x * dt;
    v *= max(0.0, 1.0 - pc.physics.w * dt);
    particles[i].prevPosition = particles[i].position;
    particles[i].position += v * dt;
    particles[i].velocity = v;
}

// ------------------------------------------------------------------ pass 2

void gridClear(uint b) {
    gridCounts[b] = 0u;
}

// atomic; buckets that overflow drop the excess particles (counted in
// gridExtra[0].x) instead of writing out of bounds.
void scatterParticle(uint i) {
    if (!isActive(i)) {
        return;
    }
    float cellSize = pc.regionMax.w;
    vec3 pos = particles[i].position;
    if (!(cellSize > 0.0) || !allFinite(pos)) {
        return; // no usable grid cell / NaN state (FINALIZE retires it)
    }

    ivec3 cell = ivec3(floor(pos / cellSize));
    uint bucket = cellHash(cell) % GRID_BUCKETS;
    uint slot = atomicAdd(gridCounts[bucket], 1u);
    if (slot < GRID_MAX_PER_BUCKET) {
        gridEntries[bucket * GRID_MAX_PER_BUCKET + slot] = i;
    } else {
        atomicAdd(gridExtra[0].x, 1u);
    }
}

// -n and -vrel (same normal speed, so the same jn) and applies the other half,
// which makes the pair momentum conserving in a purely Jacobi fashion.
void contactSolve(uint i) {
    if (!isActive(i)) {
        return;
    }
    float cellSize = pc.regionMax.w;
    if (!(cellSize > 0.0)) {
        return;
    }

    const float e = max(0.0, pc.physics.y);  // restitution (guarded: e < 0 would flip impulses)
    const float mu = max(0.0, pc.physics.z); // Coulomb coefficient (clamp() with an inverted range is undefined)
    const uint count = simParticleCount();

    vec3 pos = particles[i].position;
    vec3 vel = particles[i].velocity;
    float radius = particles[i].radius;
    if (!allFinite(pos) || !allFinite(vel)) {
        return; // NaN state: leave it to FINALIZE
    }

    uint pairContacts = 0u;
    uint terrainContacts = 0u;

    // ---- particle-particle (27 neighbouring cells of the hash grid) --------
    // Neighbours are read as-is: an invocation never writes another particle's
    // record, so the sweep is order independent (benign read/write interleaving
    // of the neighbours' own records only selects which Jacobi snapshot is seen).
    ivec3 cell = ivec3(floor(pos / cellSize));
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint bucket = cellHash(cell + ivec3(dx, dy, dz)) % GRID_BUCKETS;
                // Counts may exceed the capacity (overflowed buckets); clamp so
                // the slot reads stay inside gridEntries[].
                uint slotCount = min(gridCounts[bucket], GRID_MAX_PER_BUCKET);
                for (uint s = 0u; s < slotCount; ++s) {
                    uint j = gridEntries[bucket * GRID_MAX_PER_BUCKET + s];
                    if (j == i || j >= count) {
                        continue;
                    }

                    vec3 d = particles[j].position - pos;
                    float dist = length(d);
                    float minDist = radius + particles[j].radius;
                    if (dist >= minDist || dist <= 1e-6) {
                        continue;
                    }

                    vec3 n = d / dist; // points from i to j
                    vec3 vrel = particles[j].velocity - vel;
                    float vn = dot(vrel, n);
                    if (vn < 0.0) { // only an approaching pair exchanges impulse
                        float jn = -(1.0 + e) * vn; // >= 0
                        vel -= 0.5 * jn * n;        // half here, half in thread j

                        // Coulomb friction: the tangential impulse is the
                        // tangential relative speed, clamped to mu * |jn|. jt is
                        // negative, so -jt * t drags i along the relative
                        // tangential motion (thread j, seeing -t, gets the
                        // reaction and momentum is preserved).
                        vec3 vt = vrel - vn * n;
                        float vtl = length(vt);
                        if (vtl > 1e-6) {
                            float jt = clamp(-vtl * FRICTION_FACTOR, -mu * jn, mu * jn);
                            vel -= 0.5 * jt * (vt / vtl);
                        }
                    }

                    // Split-impulse positional separation: push i out of the
                    // overlap by half the penetration depth (the mirrored thread
                    // removes the other half). No velocity is added, so the
                    // correction cannot inject energy into a resting stack.
                    pos += (-n) * (0.5 * max(0.0, minDist - dist));
                    ++pairContacts;
                }
            }
        }
    }

    // ---- terrain SDF -------------------------------------------------------
    // Only inside the simulation region: outside it FINALIZE retires the
    // particle, and the octree lookup is not meaningful there anyway.
    if (insideRegion(pos)) {
        float sdfCellSize;
        float d = sdfAt(pos, sdfCellSize);
        if (d < radius) {
            // The gradient points from the solid (d < 0) towards the free space
            // (d > 0), i.e. it is the outward normal: the direction that
            // separates the droplet from the surface.
            vec3 n = sdfGradient(pos, sdfCellSize);
            pos += n * ((radius - d) * 0.5); // half of the penetration, soft
            float vn = dot(vel, n);
            if (vn < 0.0) {
                vel -= (1.0 + e) * vn * n; // reflect the approach speed
                vec3 vt = vel - dot(vel, n) * n;
                float vtl = length(vt);
                if (vtl > 1e-6) {
                    // Coulomb friction: never remove more tangential speed than
                    // the surface can supply (mu * |vn|), never reverse it.
                    float dvt = min(mu * abs(vn), vtl);
                    vel -= dvt * (vt / vtl);
                }
            }
            ++terrainContacts;
        }
    }

    // ---- water plane (y = regionMin.w) -------------------------------------
    float waterLevel = pc.regionMin.w;
    if (pos.y - radius < waterLevel) {
        pos.y = waterLevel + radius;
        float vn = vel.y;
        if (vn < 0.0) {
            vel.y -= (1.0 + 0.5 * e) * vn; // softer splash: half restitution
            float vtl = length(vel.xz);
            if (vtl > 1e-6) {
                float dvt = min(mu * abs(vn), vtl);
                vel.xz -= dvt * (vel.xz / vtl);
            }
        }
    }

    particles[i].position = pos;
    particles[i].velocity = vel;

    // gridExtra[0]: y = pair contacts, z = terrain contacts (+ the escaped
    // counter of pass 5, the frozen interface has no separate slot for it),
    // w = largest per-particle contact count seen this iteration.
    if (pairContacts > 0u) {
        atomicAdd(gridExtra[0].y, pairContacts);
    }
    if (terrainContacts > 0u) {
        atomicAdd(gridExtra[0].z, terrainContacts);
    }
    atomicMax(gridExtra[0].w, pairContacts + terrainContacts);
}

// margin so float noise on the boundary does not kill a valid droplet) or that
// hold a non-finite state; settle the ones that have come to rest.
void finalizeParticle(uint i) {
    if (!isActive(i)) {
        return;
    }
    vec3 pos = particles[i].position;
    vec3 vel = particles[i].velocity;
    float margin = max(0.5, particles[i].radius);

    bool escaped = any(lessThan(pos, pc.regionMin.xyz - vec3(margin)))
                || any(greaterThan(pos, pc.regionMax.xyz + vec3(margin)))
                || !allFinite(pos) || !allFinite(vel);
    if (escaped) {
        particles[i].flags = 0u;
        atomicAdd(gridExtra[0].z, 1u); // "left the region" counter
        return;
    }

    float speed = length(vel);
    if (speed < SETTLE_SPEED) {
        // Resting droplet: bleed the residual velocity strongly but keep the
        // particle active so it still renders as a settled droplet.
        particles[i].velocity = vel * max(0.0, 1.0 - SETTLE_DAMPING * pc.misc.x);
    }
}
