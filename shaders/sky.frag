#version 450

layout(location = 0) in vec3 fragPosWorld;
layout(location = 1) in vec3 fragNormal;

#include "includes/ubo.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    // Compute direction from camera to fragment (UBO exposes viewPos)
    vec3 viewDir = normalize(fragPosWorld - ubo.viewPos.xyz);
    // Use the Y component for gradient (up = 1, down = -1)
    float t = clamp(viewDir.y * 0.5 + 0.5, 0.0, 1.0);

    // Read colors from Sky UBO (set by SkyWidget)
    vec3 horizonColor = sky.skyHorizon.rgb;
    vec3 zenithColor = sky.skyZenith.rgb;

    // Warmth factor based on light elevation: when sun is low (lightDir.y near 0 or negative), increase warmth
    float userWarmth = clamp(sky.skyParams.x, 0.0, 1.0);
    // Flip lightDir.y so positive elevation corresponds to sun above the horizon
    float sunElev = -clamp(ubo.lightDir.y, -1.0, 1.0); // 1=overhead, 0=horizon, -1=below
    // compute a smooth factor in [0,1] where 1 means sun at horizon or below (warm sunsets)
    float sunFactor = clamp((1.0 - sunElev) * 0.75, 0.0, 1.0);
    sunFactor = pow(sunFactor, 1.5); // bias towards stronger effect when very low
    vec3 warmTint = vec3(1.0, 0.45, 0.2);
    horizonColor = mix(horizonColor, warmTint, sunFactor * userWarmth);

    // Slightly warm the zenith when sun is very low (creates dusk glow)
    float zenithWarm = smoothstep(0.0, 0.4, sunFactor) * 0.35 * userWarmth;
    zenithColor = mix(zenithColor, warmTint * 0.6, zenithWarm);

    // Apply exponent to control gradient falloff (allow widget override)
    float exponent = max(sky.skyParams.y, 0.01);
    // Optionally bias exponent by sun elevation so sunsets have longer transition
    exponent *= mix(1.0, 1.6, pow(sunFactor, 0.8));
    float tt = pow(t, exponent);
    vec3 dayColor = mix(horizonColor, zenithColor, tt);

    // --- Night blending ---
    // Compute a smooth day factor from sun elevation (sunElev in [-1,1])
    float dayFactor = smoothstep(-0.2, 0.2, sunElev);
    // Read night colors and intensity from Sky UBO
    vec3 nightHor = sky.nightHorizon.rgb;
    vec3 nightZen = sky.nightZenith.rgb;
    float nightIntensity = clamp(sky.nightParams.x, 0.0, 1.0);
    float starIntensity = clamp(sky.nightParams.y, 0.0, 1.0);
    vec3 nightColor = mix(nightHor, nightZen, tt);

    // Blend final color between night and day based on dayFactor
    vec3 baseColor = mix(nightColor * (1.0 - nightIntensity), dayColor, dayFactor);

    // Optional simple star effect: use small bright speckles when near full night
    float starMask = (1.0 - dayFactor) * starIntensity;
    // create a cheap pseudo-random noise from fragment position to place stars
    float starSeed = fract(sin(dot(fragPosWorld.xy ,vec2(12.9898,78.233))) * 43758.5453);
    float stars = smoothstep(0.995, 0.9995, starSeed) * starMask;
    // --- Sun flare/glow ---
    // Sun direction in world space (normalized)
    vec3 sunDir = -normalize(ubo.lightDir.xyz); // flip sun direction in all axes
    // Project sun direction into view space (from camera)
    float sunDot = dot(viewDir, sunDir);
    // Sun flare: strong when looking at sun, fades with angle
    float sunFlare = clamp(sky.skyParams.z, 0.0, 2.0); // z = user sun flare intensity
    float flare = pow(max(sunDot, 0.0), 800.0 * (1.0 - sunElev * 0.5)) * sunFlare * dayFactor;
    // Sun color: warm white, modulated by sun elevation
    vec3 sunColor = mix(vec3(1.0, 0.95, 0.8), warmTint, sunFactor * 0.5);
    // Add sun flare to color
    vec3 color = baseColor + vec3(stars) + sunColor * flare;
    outColor = vec4(color, 1.0);
}
