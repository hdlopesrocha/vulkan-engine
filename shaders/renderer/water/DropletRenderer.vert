#version 450
#extension GL_EXT_scalar_block_layout : enable

// Droplet proxy vertex shader: one unit-cube instance per simulated water
// droplet. The cube is mapped onto the droplet AABB (position +/- radius *
// 1.05) so the fragment stage ray-marches the droplet SDF sphere only for the
// pixels the sphere can cover. Drawn inside the solid color pass (dynamic
// rendering, color + depth attachments, alpha blending, depth test on /
// depth write off).
//
// FROZEN interface (host vertex layout + DropletRenderer.frag must match):
//   attributes 0/1, varyings 0/1/2, set 1 bindings 0/1, push constant PC.

#include "../../includes/SceneBindings.glsl"

layout(location = 0) out vec3 vCenter;
layout(location = 1) out float vRadius;
layout(location = 2) out vec3 vWorldPos;

// Set 1 binding 0: GPU droplet simulation state (scalar layout: 48 B stride —
// position 0, radius 12, velocity 16, mass 28, prevPosition 32, flags 44).
struct Droplet {
    vec3 position;
    float radius;
    vec3 velocity;
    float mass;
    vec3 prevPosition;
    uint flags;
};
layout(std430, set = 1, binding = 0, scalar) readonly buffer DropletState {
    Droplet droplets[];
};

// params = (particleCount, radiusScale, ior, debugMode)
layout(push_constant) uniform PC {
    vec4 params;
} pc;

// Camera-facing billboard: a square of half-extent radius*1.05 around the
// droplet center (the sphere always projects inside it). Non-indexed 6-vertex
// draw; the camera basis comes from the inverse view-projection so the proxy
// needs no vertex buffer and no cull-mode guesswork — every covered pixel is
// shaded exactly once.
const vec2 kQuad[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

void main() {
    uint idx = uint(gl_InstanceIndex);
    // particleCount caps the drawn prefix (0 = use the whole SSBO); the
    // per-droplet flags bit0 is the authoritative active marker.
    uint count = (pc.params.x > 0.5)
        ? min(uint(pc.params.x + 0.5), uint(droplets.length()))
        : uint(droplets.length());
    if (idx >= count || (droplets[idx].flags & 1u) == 0u) {
        // Inactive/overflow instance: degenerate clip position (w = 0) so the
        // triangle is culled and the fragment stage never runs.
        gl_Position = vec4(0.0, 0.0, -1.0, 0.0);
        vCenter = vec3(0.0);
        vRadius = 0.0;
        vWorldPos = vec3(0.0);
        return;
    }

    Droplet d = droplets[idx];
    float radius = max(d.radius * max(pc.params.y, 0.0), 0.0);
    vec3 center = d.position;

    // Camera basis (world space) from the inverse view-projection: the columns
    // of the inverse linear part are the world directions of NDC +X / +Y.
    vec3 right = normalize((ubo.invViewProjection * vec4(1.0, 0.0, 0.0, 0.0)).xyz);
    vec3 up = normalize((ubo.invViewProjection * vec4(0.0, 1.0, 0.0, 0.0)).xyz);

    vec2 q = kQuad[gl_VertexIndex % 6] * 2.0 - 1.0;
    vec3 worldPos = center + (right * q.x + up * q.y) * (radius * 1.05);

    vCenter = center;
    vRadius = radius;
    vWorldPos = worldPos;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
}
