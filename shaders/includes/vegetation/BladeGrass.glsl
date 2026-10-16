// Bladed-grass shared helpers (triangle tufts, no billboards, no SDF).
// Requires nothing except a hash; wind/tilt helpers come from
// VegetationCommon.glsl / WindField.glsl included separately.
#ifndef BLADE_GRASS_GLSL
#define BLADE_GRASS_GLSL

// Blades per tuft (one triangle per blade: left/right base + tip).
// Must match RendererUtils::kBladeCount (36 verts = 12 blades x 3).
#define BLADE_COUNT 12

// Deterministic per-blade hash — canonical in the blade index ONLY.
//
// This MUST NOT depend on the instance world position: the impostor capture
// renders the tuft at the origin and bakes that exact blade layout, normals
// and colors into the impostor arrays. Every runtime tuft therefore has to
// use the same canonical layout, or the impostor shown past the hand-off
// distance would be a different plant than the blades drawn just before it
// (different blade directions/heights/widths -> different normal colors at
// the same world spot). Per-instance variety comes from the instance
// rotation (rotFrac), the baked height scale and the biome palette.
// Returns 4 uncorrelated [0,1) values without trig.
vec4 bladeHash(float bladeIdx) {
    vec3 p = vec3(bladeIdx * 17.13, bladeIdx * 9.71, bladeIdx * 5.37);
    vec3 h = fract(sin(vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                             dot(p, vec3(269.5, 183.3, 246.1)),
                             dot(p, vec3(113.5, 271.9, 124.6)))) * 43758.5453);
    return vec4(h, fract(h.x * 7.13 + h.y * 3.71 + bladeIdx * 0.317));
}

// Blade palette per grass variant (0=foliage/meadow, 1=grass/lawn, 2=wild/dry).
// Root (dark, soil-shaded) and tip (sun-lit) colors; frag lerps by heightT.
void bladePalette(int variant, out vec3 rootColor, out vec3 tipColor) {
    if (variant == 0) {
        rootColor = vec3(0.075, 0.21, 0.045);
        tipColor  = vec3(0.32, 0.55, 0.16);
    } else if (variant == 2) {
        rootColor = vec3(0.16, 0.14, 0.05);
        tipColor  = vec3(0.55, 0.47, 0.22);
    } else {
        rootColor = vec3(0.06, 0.17, 0.05);
        tipColor  = vec3(0.38, 0.60, 0.20);
    }
}

#endif // BLADE_GRASS_GLSL
