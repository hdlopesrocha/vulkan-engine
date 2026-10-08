#ifndef SDF_ROCK_GLSL
#define SDF_ROCK_GLSL

// Rock boulder primitive: static Perlin-displaced sphere. Requires
// includes/sdf/SdfNoise.glsl (sdfFbm) included first; the CPU evaluator
// branch lives in SdfRenderer.frag (def.prim == SDF_PRIM_ROCK), mirroring
// the Smoke special case.
//
// p0.x = base radius (local units), p0.y = noise frequency (per local unit),
// p0.z = displacement amplitude (fraction of the radius). seed offsets the
// noise lattice per instance so every boulder is unique (SdfInstance::seed).
//
// Returns a conservative lower bound of the true distance (divided by an
// analytic Lipschitz bound of the fBm displacement), so sphere tracing can
// never overstep the noisy surface while the zero set stays unchanged.
float sdRock(vec3 p, vec4 p0, float seed) {
    float r = max(p0.x, 1e-4);
    float ns = max(p0.y, 1e-5);
    float amp = max(p0.z, 0.0);
    vec3 q = p * ns + vec3(seed * 37.0, seed * 17.0, seed * 53.0);
    float n = sdfFbm(q); // [0, 1]
    float disp = (n - 0.5) * 2.0 * amp * r;
    // fBm gradient bound: the normalized 3-octave sum with value-noise
    // quintic fade has |grad| <= ~3.2*ns; times the 2*amp*r displacement
    // range gives the factor below.
    float lip = 1.0 + 6.5 * amp * ns * r;
    return (length(p) - (r + disp)) / max(lip, 1.0);
}

#endif // SDF_ROCK_GLSL
