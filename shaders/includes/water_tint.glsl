#ifndef WATER_TINT_GLSL
#define WATER_TINT_GLSL

// Water tint: ONE region, so this is a single colour. `depth` is kept in the
// signature because the callers key the shore fade and the tint blend to it,
// but there is no depth-band palette any more.
vec3 waterRegionTint(in WaterParamsNamed wp, float depth) {
    return wp.waterColor;
}

#endif // WATER_TINT_GLSL
