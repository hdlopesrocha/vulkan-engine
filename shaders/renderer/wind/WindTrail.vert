#version 450

// Wind-trail geometry expansion (WindRenderer).
//
// No vertex buffer: the whole draw is generated from gl_VertexIndex. The CPU
// issues (particleCount * segments * 6) vertices, and this shader maps each one
// to (particle, trail segment, quad corner). The segment is expanded into a
// SCREEN-SPACE oriented quad, which gives real pixel-width trails on every
// vendor (the wideLines feature is not required) plus a soft lateral falloff.
//
// Trail samples are read from the ring buffer written by
// shaders/computation/wind/WindParticles.comp. A sample's age is
// (wind clock - sample time), so the fade is continuous and independent of the
// trail's storage order; segments older than the configured duration are
// clipped (all corners pushed outside depth, which the rasterizer discards).

#include "../../includes/wind/WindParticle.glsl"
#include "../../ubo/WindParticlesUBO.glsl"

layout(set = 0, binding = 0, std140) uniform WindParticlesBlock {
    WindParticlesUBO wp;
};
layout(set = 0, binding = 1, std430) readonly buffer Particles {
    WindParticle particles[];
};
layout(set = 0, binding = 2, std430) readonly buffer TrailSamples {
    vec4 trailSamples[];
};
layout(set = 0, binding = 3, std430) readonly buffer TrailSpeeds {
    float trailSpeeds[];
};

layout(location = 0) out float vLateral; // -1..+1 across the trail width
layout(location = 1) out float vFade;    // age/spawn fade for this endpoint
layout(location = 2) out float vSpeed;   // local wind speed (m/s) at this endpoint

// Corners of the two triangles that make a segment quad:
//   tri 0 = (a-, a+, b-), tri 1 = (a+, b+, b-)
const float kSide[6] = float[6](-1.0, 1.0, -1.0, 1.0, 1.0, -1.0);
const float kEnd[6]  = float[6](0.0, 0.0, 1.0, 0.0, 1.0, 1.0);

void main() {
    vLateral = 0.0;
    vFade = 0.0;
    vSpeed = 0.0;
    // Outside Vulkan clip depth (z < 0): the triangle is discarded.
    gl_Position = vec4(0.0, 0.0, -1.0, 1.0);

    uint particleCount = uint(wp.timing.z + 0.5);
    uint segments = uint(clamp(ceil(wp.trail.x / max(wp.trail.y, 1e-4)),
                               1.0, max(wp.trail.z - 1.0, 1.0)));
    uint vertsPerParticle = segments * 6u;
    if (particleCount == 0u || vertsPerParticle == 0u) return;

    uint pid = uint(gl_VertexIndex) / vertsPerParticle;
    if (pid >= particleCount) return;
    uint local = uint(gl_VertexIndex) % vertsPerParticle;
    uint seg = local / 6u;
    uint corner = local % 6u;

    WindParticle particle = particles[pid];
    uint head = particle.trail.x;
    uint count = particle.trail.y;
    // Segments walk backwards in time from the head; a segment needs two
    // stored samples.
    if (count < 2u || seg + 1u >= count) return;

    uint capacity = uint(WIND_TRAIL_CAPACITY);
    vec4 newer = trailSamples[pid * capacity + (head + capacity - seg) % capacity];
    vec4 older = trailSamples[pid * capacity + (head + capacity - seg - 1u) % capacity];

    float duration = max(wp.trail.x, 1e-3);
    float ageNewer = max(wp.timing.x - newer.w, 0.0);
    float ageOlder = max(wp.timing.x - older.w, 0.0);
    if (ageOlder > duration) return; // whole segment expired

    float fadeNewer = clamp(1.0 - ageNewer / duration, 0.0, 1.0);
    float fadeOlder = clamp(1.0 - ageOlder / duration, 0.0, 1.0);
    // Fade particles in after a (re)spawn so a new trail does not pop.
    float spawnFade = smoothstep(0.0, max(wp.extra.y, 1e-3), particle.position.w);

    vec4 clipNewer = wp.viewProjection * vec4(newer.xyz, 1.0);
    vec4 clipOlder = wp.viewProjection * vec4(older.xyz, 1.0);
    if (clipNewer.w <= 1e-4 || clipOlder.w <= 1e-4) return; // behind the eye

    // Screen-space direction -> unit perpendicular for the quad width. The
    // viewport size converts NDC <-> pixels (NDC spans 2.0 per axis, so a
    // half-width of w/2 px is w/screen in NDC).
    vec2 screen = max(wp.spawn.zw, vec2(1.0));
    vec2 dir = (clipOlder.xy / clipOlder.w - clipNewer.xy / clipNewer.w) * screen;
    float len = length(dir);
    dir = (len > 1e-5) ? dir / len : vec2(1.0, 0.0);
    vec2 perp = vec2(-dir.y, dir.x);

    float endSel = kEnd[corner];
    float side = kSide[corner];
    vec4 base = (endSel < 0.5) ? clipNewer : clipOlder;
    vec2 offset = perp * (max(wp.trail.w, 0.1) / screen) * side;

    gl_Position = base;
    gl_Position.xy += offset * base.w;

    vLateral = side;
    vFade = ((endSel < 0.5) ? fadeNewer : fadeOlder) * spawnFade;
    // Speed of the wind at this endpoint (stored with the sample by the
    // simulation) drives the blue->green->yellow->red ramp in the fragment
    // stage.
    vSpeed = trailSpeeds[pid * capacity + ((endSel < 0.5) ? (head + capacity - seg) % capacity
                                                          : (head + capacity - seg - 1u) % capacity)];
}
