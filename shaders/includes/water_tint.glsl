#ifndef WATER_TINT_GLSL
#define WATER_TINT_GLSL

// Depth-region water tint ramp, shared by the raster water surface
// (water_surface.glsl) and water surfaces seen inside RT rays
// (rt_reflection.glsl) so both shade with the same colors.
//
// The ramp is a 5-stop color map keyed to the measured water DEPTH and the
// per-layer shore-wave zone boundaries:
//
//   depth 0            shore line        regionShoreColor
//   zoneShallowDepth   foam decay band   regionShallowColor
//   zoneBreakDepth     breaker line      regionBreakerColor
//   mid(break, deep)   shoaling band     regionShoalColor
//   zoneDeepDepth      open ocean        regionDeepColor
//
// Neighboring stops blend through a smoothstep window whose half-width is
// `regionTintParams.y` (softness) times the adjacent stop spacing, so 0
// gives hard region edges and 0.5 a fully soft ramp. Requires ubo.glsl
// (WaterParamsGPU declaration) to be included first.
vec3 waterRegionTint(in WaterParamsGPU wp, float depth) {
    float zDeep = max(wp.waveZones.x, 1.0);
    float zBreak = clamp(wp.waveZones.y, 1e-3, zDeep);
    float zShallow = clamp(wp.waveZones.z, 0.0, zBreak);
    float soft = clamp(wp.regionTintParams.y, 0.0, 0.5);

    float s1 = zShallow;
    float s2 = zBreak;
    float s3 = mix(zBreak, zDeep, 0.5);
    float s4 = zDeep;

    float w1 = soft * max(s1, 1e-3);
    float w2 = soft * max(min(s2 - s1, s3 - s2), 1e-3);
    float w3 = soft * max(min(s3 - s2, s4 - s3), 1e-3);
    float w4 = soft * max(s4 - s3, 1e-3);

    vec3 col = wp.regionShoreColor.rgb;
    col = mix(col, wp.regionShallowColor.rgb,
              smoothstep(max(s1 - w1, 0.0), s1 + w1, depth));
    col = mix(col, wp.regionBreakerColor.rgb,
              smoothstep(s2 - w2, s2 + w2, depth));
    col = mix(col, wp.regionShoalColor.rgb,
              smoothstep(s3 - w3, s3 + w3, depth));
    col = mix(col, wp.regionDeepColor.rgb,
              smoothstep(s4 - w4, s4 + w4, depth));
    return col;
}

#endif // WATER_TINT_GLSL
