#version 450
#extension GL_EXT_scalar_block_layout : enable

// Droplet fragment stage: analytic ray/sphere entry/exit bounds, then a short
// SDF sphere trace to the surface, then water-like shading — Fresnel mix of a
// blurred refracted sky tap and a sharp reflected sky tap (plus one extra
// through-sphere lens tap) and a subtle sun highlight. Coverage alpha feathers
// the silhouette (~1.5 px); the color is premultiplied (the pass blends with
// alpha blending, depth test on, depth write off). gl_FragDepth carries the
// exact hit depth.
//
// FROZEN interface: varyings 0/1/2, set 1 bindings 0/1, push constant PC —
// must match DropletRenderer.vert.

#include "../../includes/SceneBindings.glsl"

layout(location = 0) in vec3 vCenter;
layout(location = 1) in float vRadius;
layout(location = 2) in vec3 vWorldPos;

// Set 1 binding 0: droplet state, identical layout to the vertex stage.
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

// Set 1 binding 1: sky/environment color, sampled with the water pass's
// direction->equirect mapping.
layout(set = 1, binding = 1) uniform sampler2D dropletSkyTex;

// params = (particleCount, radiusScale, ior, debugMode)
layout(push_constant) uniform PC {
    vec4 params;
} pc;

layout(location = 0) out vec4 outColor;

// Direction -> equirect UV (copy of the water/RT sky sampling helper).
vec2 dropletDirToEquirectUV(vec3 dir) {
    const float PI = 3.14159265358979;
    return vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
                acos(clamp(dir.y, -1.0, 1.0)) / PI);
}

void main() {
    vec3 ro = ubo.viewPosition;
    // Solid-pass convention: the rasterized proxy cube interpolates its world
    // position, so the camera ray through this pixel is exact (this pass has no
    // screen-size uniform to rebuild the ray from gl_FragCoord).
    vec3 rd = normalize(vWorldPos - ro);

    vec3 center = vCenter;
    float radius = max(vRadius, 1e-5);

    // Silhouette coverage: the perpendicular ray-to-center distance,
    // differentiated in screen space. alpha = 1 at the sphere, feathered to 0
    // within ~1.5 px of the silhouette; outside it the pixel contributes
    // nothing (ray misses the sphere).
    float edge = radius - length(cross(rd, center - ro));
    float feather = max(fwidth(edge) * 1.5, 1e-6);
    float alpha = smoothstep(0.0, 1.0, edge / feather);
    if (alpha <= 0.0) discard;

    // Analytic ray/sphere entry/exit (bounds the SDF trace below).
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc <= 0.0) discard;
    float sq = sqrt(disc);
    float t0 = max(-b - sq, 0.0);
    float t1 = -b + sq;
    if (t1 <= 0.0) discard; // sphere entirely behind the camera

    // Short SDF sphere trace from the analytic entry to the surface. |sdf|
    // keeps the trace progressing when the start is inside (camera inside the
    // droplet marches out to the far wall); grazing rays that exhaust the step
    // budget fall back to the analytic root.
    float eps = max(radius * 1e-3, 1e-4);
    float t = t0;
    bool converged = false;
    int steps = 0;
    for (int i = 0; i < 32; ++i) {
        steps = i + 1;
        float sdf = length(ro + rd * t - center) - radius;
        if (abs(sdf) < eps) { converged = true; break; }
        t += max(0.5 * abs(sdf), 1e-3);
        if (t > t1) break;
    }
    float tHit = converged ? t : ((t0 > 0.0) ? t0 : t1);

    vec3 hit = ro + rd * tHit;
    vec3 n = normalize(hit - center);
    if (dot(n, rd) > 0.0) n = -n; // face the viewer (inside/fallback hits)
    vec3 v = rd;                  // incident direction, camera -> hit

    // Fresnel (Schlick, F0 = 0.02 for water).
    float cosTheta = clamp(dot(n, -v), 0.0, 1.0);
    float fresnel = 0.02 + 0.98 * pow(1.0 - cosTheta, 5.0);

    // Reflection: sharp sky tap.
    vec3 reflDir = reflect(v, n);
    vec3 reflCol = textureLod(dropletSkyTex, dropletDirToEquirectUV(reflDir), 0.0).rgb;

    // Refraction: blurred entry tap; one extra tap through the sphere's exit
    // point adds the lens bend. Total: 2 refracted + 1 reflected sky taps.
    float ior = max(pc.params.z, 1.0);
    vec3 refrCol;
    vec3 refrDir = refract(v, n, 1.0 / ior);
    if (dot(refrDir, refrDir) < 1e-8) {
        refrCol = reflCol; // total internal reflection
    } else {
        refrCol = textureLod(dropletSkyTex, dropletDirToEquirectUV(refrDir), 1.0).rgb;
        vec3 ocR = hit - center;
        float bR = dot(ocR, refrDir);
        float cR = dot(ocR, ocR) - radius * radius;
        float discR = bR * bR - cR;
        if (discR > 0.0) {
            vec3 exitP = hit + refrDir * (-bR + sqrt(discR));
            vec3 exitN = normalize(exitP - center);
            vec3 outDir = refract(refrDir, -exitN, ior); // water -> air
            if (dot(outDir, outDir) > 1e-8) {
                vec3 lensCol = textureLod(dropletSkyTex, dropletDirToEquirectUV(outDir), 1.0).rgb;
                refrCol = mix(refrCol, lensCol, 0.5);
            }
        }
    }

    vec3 color = mix(refrCol, reflCol, fresnel);

    // Subtle sun highlight (half vector between the to-sun and to-camera dirs).
    vec3 toSun = -normalize(ubo.lightDirection);
    vec3 hv = toSun - v;
    float hl = length(hv);
    if (hl > 1e-4) {
        color += ubo.lightColor * 0.35 * pow(max(dot(n, hv / hl), 0.0), 96.0);
    }

    // Debug views: 1 = flat normal, 2 = march step count heat (blue -> red).
    if (pc.params.w > 1.5) {
        float heat = float(steps) / 32.0;
        color = vec3(heat, 0.25 * heat, 1.0 - heat);
    } else if (pc.params.w > 0.5) {
        color = n * 0.5 + 0.5;
    }

    // Exact hit depth (the projection already carries the Vulkan Y flip;
    // z is in [0,1]).
    vec4 clip = ubo.viewProjection * vec4(hit, 1.0);
    gl_FragDepth = (clip.w > 1e-6) ? clamp(clip.z / clip.w, 0.0, 1.0) : 1.0;

    outColor = vec4(color * alpha, alpha); // premultiplied
}
