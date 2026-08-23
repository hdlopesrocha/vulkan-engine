// Shared declarations for all ray-tracing shaders. Reuses the project's existing
// UBO / material / sky definitions (set=0) and adds the TLAS (binding 20) plus
// the per-workload output storage image (set=1). No scene data is duplicated.

#include "ubo.glsl"

// Unified TLAS (built by RayTracingRenderer from per-chunk BLASes).
layout(set = 0, binding = 20) uniform accelerationStructureEXT tlas;

// Geometry-kind constants, matching GeometryKind in RayTracingRenderer.hpp. The
// low byte of an instance's custom index carries the kind; vegetation also packs
// its foliage array layer into the next byte.
const uint RT_KIND_SOLID = 0u;
const uint RT_KIND_WATER = 1u;
const uint RT_KIND_VEG   = 2u;

// Vegetation leaf-opacity texture (sampler2DArray), bound by RayTracingRenderer
// at set=0 binding 14. Used by the closest-hit shader to alpha-test vegetation
// for shadow rays (transparent leaf texels do not occlude).
layout(set = 0, binding = 14) uniform sampler2DArray rtOpacity;

// Output storage image (written by the raygen shader of each workload). The
// format qualifier is declared per-workload .rgen file because the shadow mask
// uses a single-channel r32f image while reflection/refraction use rgba32f.
// shadow.rgen / reflect.rgen / refract.rgen each declare `outImage` themselves.

// Ray payload shared by every workload. `hit` distinguishes miss (0) from a
// geometry intersection so shadow rays can compute occlusion cheaply, while
// `color`/`normal`/`pos` carry shading data for reflection/refraction.
// `rayType` carries the WORKLOAD kind for the primary render pass (which needs
// shadow + reflection + refraction secondary rays in a single closest-hit):
//   0 = primary (full shading), 1 = shadow, 2 = reflection, 3 = refraction.
struct Payload {
    vec3  color;
    float hit;     // 1.0 when a triangle was hit, 0.0 on miss
    vec3  normal;
    vec3  pos;
    uint  rayType;
};

// Workload mode pushed from the CPU: 0 = shadow, 1 = reflection, 2 = refraction.
layout(push_constant) uniform PC {
    uint uMode;
} pc;

// Cheap sky/environment color from a world-space direction, reusing the sky UBO
// already bound at set=0 binding=6. Used by miss shaders and as the fallback for
// reflection/refraction rays that escape the scene.
vec3 environmentColor(vec3 dir) {
    float t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 horizon = sky.skyHorizon.rgb;
    vec3 zenith = sky.skyZenith.rgb;
    // Blend night colors when night intensity is set.
    horizon = mix(horizon, sky.nightHorizon.rgb, sky.nightParams.x);
    zenith = mix(zenith, sky.nightZenith.rgb, sky.nightParams.x);
    return mix(horizon, zenith, pow(t, max(sky.skyParams.y, 0.5)));
}

// Camera ray for the current launch pixel, reconstructed from the inverse view
// projection in the scene UBO.
void computeCameraRay(out vec3 origin, out vec3 dir) {
    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);
    vec4 nearH = ubo.invViewProjection * vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    vec4 farH  = ubo.invViewProjection * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 nearP = nearH.xyz / nearH.w;
    vec3 farP  = farH.xyz / farH.w;
    origin = ubo.viewPos.xyz;
    dir = normalize(farP - nearP);
}
