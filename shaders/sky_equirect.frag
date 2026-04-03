#version 450

// Sky equirectangular map renderer.
// Renders the procedural sky dome to an equirectangular 2D texture.
// UV.x = longitude [0,2π], UV.y = latitude [0,π] (top = zenith, bottom = nadir).
// This texture is view-independent and can be sampled by any shader using
// a 3D direction converted to equirect UV coordinates.

#include "includes/ubo.glsl"

// Push constant with equirect target resolution
layout(push_constant) uniform PushConstants {
    vec2 resolution; // width, height of equirect target
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

void main() {
    // Convert fragment position to equirect UV [0,1]
    vec2 uv = gl_FragCoord.xy / pc.resolution;

    // Convert UV to spherical direction
    // theta: polar angle from zenith (0 at top, PI at bottom)
    // phi: azimuthal angle (-PI to PI)
    // The inverse mapping (direction → UV) used by water.frag and postprocess.frag is:
    //   uv.x = atan(z, x) / (2*PI) + 0.5
    // So the matching forward mapping must be:
    //   phi = (uv.x - 0.5) * 2*PI
    float theta = uv.y * PI;
    float phi   = (uv.x - 0.5) * 2.0 * PI;

    // Spherical to Cartesian (Y-up)
    vec3 dir;
    dir.x = sin(theta) * cos(phi);
    dir.y = cos(theta);              // 1 at zenith (top), -1 at nadir (bottom)
    dir.z = sin(theta) * sin(phi);

    // === SKY COLOR (same logic as sky.frag) ===
    vec3 viewDir = dir;
    float t = clamp(viewDir.y * 0.5 + 0.5, 0.0, 1.0);

    // Read colors from Sky UBO
    vec3 horizonColor = sky.skyHorizon.rgb;
    vec3 zenithColor  = sky.skyZenith.rgb;

    // Warmth factor based on light elevation
    float userWarmth = clamp(sky.skyParams.x, 0.0, 1.0);
    float sunElev = -clamp(ubo.lightDir.y, -1.0, 1.0);
    float sunFactor = clamp((1.0 - sunElev) * 0.75, 0.0, 1.0);
    sunFactor = pow(sunFactor, 1.5);
    vec3 warmTint = vec3(1.0, 0.45, 0.2);
    horizonColor = mix(horizonColor, warmTint, sunFactor * userWarmth);

    // Slightly warm the zenith when sun is very low
    float zenithWarm = smoothstep(0.0, 0.4, sunFactor) * 0.35 * userWarmth;
    zenithColor = mix(zenithColor, warmTint * 0.6, zenithWarm);

    // Apply exponent to control gradient falloff
    float exponent = max(sky.skyParams.y, 0.01);
    exponent *= mix(1.0, 1.6, pow(sunFactor, 0.8));
    float tt = pow(t, exponent);
    vec3 dayColor = mix(horizonColor, zenithColor, tt);

    // --- Night blending ---
    float dayFactor = smoothstep(-0.2, 0.2, sunElev);
    vec3 nightHor = sky.nightHorizon.rgb;
    vec3 nightZen = sky.nightZenith.rgb;
    float nightIntensity = clamp(sky.nightParams.x, 0.0, 1.0);
    float starIntensity  = clamp(sky.nightParams.y, 0.0, 1.0);
    vec3 nightColor = mix(nightHor, nightZen, tt);

    vec3 baseColor = mix(nightColor * (1.0 - nightIntensity), dayColor, dayFactor);

    // Star effect — use direction-based hash for stable positions
    float starMask = (1.0 - dayFactor) * starIntensity;
    float starSeed = fract(sin(dot(dir.xz * 1000.0, vec2(12.9898, 78.233))) * 43758.5453);
    float stars = smoothstep(0.995, 0.9995, starSeed) * starMask;

    // --- Sun flare/glow ---
    vec3 sunDir = -normalize(ubo.lightDir.xyz);
    float sunDot = dot(viewDir, sunDir);
    float sunFlare = clamp(sky.skyParams.z, 0.0, 2.0);
    float flare = pow(max(sunDot, 0.0), 800.0 * (1.0 - sunElev * 0.5)) * sunFlare * dayFactor;
    vec3 sunColor = mix(vec3(1.0, 0.95, 0.8), warmTint, sunFactor * 0.5);

    vec3 color = baseColor + vec3(stars) + sunColor * flare;
    outColor = vec4(color, 1.0);
}
