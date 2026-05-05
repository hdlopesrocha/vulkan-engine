// Common helpers for vegetation and impostor geometry shaders.
// Requires perlin2d.glsl and the push_constant block (densityParams,
// cameraPosAndFalloff) to be declared before this include.

// Rotate a direction vector around the world-up (Y) axis.
vec3 rotateY(vec3 v, float c, float s) {
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

// Per-instance height variation driven by 2D Perlin noise on world XZ.
// Returns a scale in [0.6, 1.4] that should be applied to billboardScale.
float vegetationHeightScale(vec2 worldXZ) {
    float n = perlin2(worldXZ * 0.008);
    return 0.6 + 0.8 * (n * 0.5 + 0.5);
}

// Distance-based density thinning shared by vegetation.geom and impostors.geom.
// Requires push constants: densityParams, cameraPosAndFalloff.
float densityFactorForDistance(float distanceToCamera) {
    if (densityParams.x < 0.5) return 1.0;

    float nearDistance = max(0.0, densityParams.y);
    float minFactor    = clamp(densityParams.w, 0.0, 1.0);
    float falloff      = cameraPosAndFalloff.w;
    if (distanceToCamera <= nearDistance || minFactor >= 1.0 || falloff <= 0.0) {
        return 1.0;
    }

    float density = exp(-falloff * (distanceToCamera - nearDistance));
    return clamp(density, minFactor, 1.0);
}
