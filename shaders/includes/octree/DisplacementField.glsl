#ifndef OCTREE_DISPLACEMENT_FIELD_GLSL
#define OCTREE_DISPLACEMENT_FIELD_GLSL

// Shared true-displacement field definition for the octree raymarcher.
//
// Both the camera march (OctreeRaymarch.frag) and the CSM terrain caster
// (OctreeRaymarchShadow.frag) include this file, so the surface they intersect
// is defined by exactly the same expression:
//
//     F(p) = d_original(p) - h(p)
//
// `h(p)` is a signed world-space offset: positive moves the isosurface along
// the base outward normal (the raster tessellation convention), negative
// recesses it. The displaced surface is the zero level set of F. The base
// field is the octree's trilinear reconstruction of the stored corner SDFs;
// the displacement is sampled in world space from the material height map,
// using the same per-vertex mapping mode / UV scale the material shading uses.
//
// Includers must provide:
//   * SceneBindings.glsl (materials[], materialNamed(), brushLayer(), ubo)
//   * Textures.glsl      (heightArray)
//   * Common.glsl        (legacy sampleHeight* helpers)
//   * TextureMapping.glsl
//
// GLSL note: this header is deliberately free of control-flow-dependent
// implicit-LOD texture() calls; every fetch uses textureLod() so it is valid in
// the divergent raymarch loop and in the shadow caster's non-uniform branch.

#include "../TextureMapping.glsl"
#include "../noise/Perlin2D.glsl"
#include "../noise/Perlin.glsl"
#include "../sdf/SdfNoise.glsl"
#include "../../types/MaterialNamed.glsl"

// Resolved displacement configuration for one material.
struct RmDisplacement {
    bool enabled;       // a non-zero offset can be produced
    bool legacy;       // old tessellation-equivalent path
    int material;      // owning material index (invert flags / own layer)
    int textureIndex;  // brush index for the height layer, -1 = own
    float amplitude;   // signed world units at normalized extremes
    float frequency;   // UV multiplier on top of the per-node scale
    float offset;      // normalized height bias
    float strength;    // [0..1]
    float bounds;      // |offset| clamp, 0 = |amplitude| * strength
    float maxAbs;      // conservative |h| bound for this material
    int pattern;       // 0 = texture, 1 = triangular scales, 2 = hexagons
    float patternDisplacement;
    float patternRepeat;
    float patternTime;
    vec3 patternVelocity;
    float patternIntensity;
};

// Resolve the material settings. The legacy branch reproduces the previous
// tessellation bump exactly (mappingEnabled + global tessellation, half-scale
// bound, offset 0); the new branch is independent of tessellation.
RmDisplacement rmDisplacementParams(int material) {
    RmDisplacement d;
    d.enabled = false;
    d.legacy = false;
    d.material = material;
    d.textureIndex = -1;
    d.amplitude = 0.0;
    d.frequency = 1.0;
    d.offset = 0.0;
    d.strength = 0.0;
    d.bounds = 0.0;
    d.maxAbs = 0.0;
    d.pattern = 0;
    d.patternDisplacement = 0.0;
    d.patternRepeat = 1.0;
    d.patternTime = 0.0;
    d.patternVelocity = vec3(0.0, 1.0, 0.0);
    d.patternIntensity = 1.0;
    if (material < 0) return d;

    MaterialNamed m = materialNamed(materials[material]);

    // The new material path takes precedence when configured; otherwise keep
    // the legacy tessellation bump bit-compatible. A secondary SDF pattern is
    // an independent source (it does not need a height texture amplitude).
    const bool textureConfigured = m.displacementStrength > 0.0 &&
                                   m.displacementAmplitude != 0.0;
    const bool patternConfigured = m.displacementStrength > 0.0 &&
                                   m.displacementPattern > 0 &&
                                   m.displacementPatternDisplacement != 0.0;
    const bool newConfigured = ubo.displacementEnabled != 0u &&
                               (textureConfigured || patternConfigured);
    if (!newConfigured) {
        if (m.mappingEnabled && ubo.tessellationEnabled != 0u) {
            d.enabled = true;
            d.legacy = true;
            d.amplitude = m.tessHeightScale;
            d.frequency = 1.0;
            d.offset = 0.0;
            d.strength = 1.0;
            d.bounds = abs(m.tessHeightScale);
            d.textureIndex = -1;
            // h is clamped to [0,1] by the sampler, so |h - 0.5| <= 0.5.
            d.maxAbs = 0.5 * abs(m.tessHeightScale);
        }
        return d;
    }

    d.strength = clamp(m.displacementStrength, 0.0, 1.0);
    d.pattern = clamp(m.displacementPattern, 0, 9);
    d.patternDisplacement = m.displacementPatternDisplacement;
    d.patternRepeat = max(m.displacementPatternRepeat, 1.0e-4);
    d.patternTime = m.displacementSegmentTime;
    d.patternVelocity = m.displacementSegmentVelocity;
    d.patternIntensity = clamp(m.displacementSegmentIntensity, 0.0, 1.0);
    if (d.pattern > 0) {
        d.enabled = abs(d.patternDisplacement) * d.strength * d.patternIntensity > 0.0;
        d.amplitude = d.patternDisplacement;
        d.frequency = 1.0;
        d.offset = 0.0;
        d.bounds = abs(d.patternDisplacement) * d.strength * d.patternIntensity;
        d.textureIndex = -1;
        d.maxAbs = d.bounds;
        return d;
    }

    d.enabled = true;
    d.amplitude = m.displacementAmplitude;
    d.frequency = max(m.displacementFrequency, 1e-4);
    d.offset = m.displacementOffset;
    d.bounds = max(m.displacementBounds, 0.0);
    d.textureIndex = m.displacementTexture;
    // Normalized height range after the offset. |h - 0.5 + offset| <= nmax.
    float nmax = max(abs(-0.5 + d.offset), abs(0.5 + d.offset));
    float configured = d.bounds > 0.0 ? d.bounds : abs(d.amplitude) * d.strength;
    d.maxAbs = min(configured, abs(d.amplitude) * d.strength * nmax);
    return d;
}

bool rmDisplacementActive(int material) {
    return rmDisplacementParams(material).enabled;
}

float rmDisplacementBound(int material) {
    return rmDisplacementParams(material).maxAbs;
}

// Height fetch with explicit LOD. The owner material supplies the invert
// flags; `layer` is the resolved texture-array layer.
float rmHeightSampleLod(vec2 uv, int layer, bool invertHeight, bool invertWidth,
                        float lod) {
    vec2 tc = uv;
    if (invertHeight) tc.y = 1.0 - tc.y;
    if (invertWidth) tc.x = 1.0 - tc.x;
    return clamp(textureLod(heightArray, vec3(tc, float(layer)), lod).r, 0.0, 1.0);
}

// Sampled height for the new path. Triplanar uses the same world-space
// projection/sign convention as Common.glsl; the three heights are blended
// with the geometric-normal weights so transitions stay continuous. Single
// projection / angular / custom modes use the dominant mapped UV.
float rmSampleDisplacementHeight(RmDisplacement d, MaterialNamed m, int texType,
                                 vec2 uvScale, vec3 p, vec3 n, float lod) {
    vec2 scale = uvScale * d.frequency;
    vec2 uvX, uvY, uvZ;
    vec3 w;
    computeMaterialUVs(texType, d.material, p, n, scale,
                       ubo.triplanarThreshold, ubo.triplanarExponent,
                       uvX, uvY, uvZ, w);

    // Procedural displacement source: a world-space 2D Perlin field in the
    // same projection frame as the material mapping. This is the "procedural
    // displacement function" path; no height texture is sampled.
    if (d.textureIndex == -2) {
        if (mappingUsesTriplanar(texType)) {
            float nX = perlin2(uvX) * w.x;
            float nY = perlin2(uvY) * w.y;
            float nZ = perlin2(uvZ) * w.z;
            return clamp(0.5 + 0.5 * (nX + nY + nZ), 0.0, 1.0);
        }
        float n = perlin2(dominantMappingUV(w, uvX, uvY, uvZ));
        return clamp(0.5 + 0.5 * n, 0.0, 1.0);
    }

    int layer = brushLayer(max(d.textureIndex >= 0 ? d.textureIndex : d.material, 0));
    if (mappingUsesTriplanar(texType)) {
        float hX = w.x > 0.0 ? rmHeightSampleLod(uvX, layer, m.invertHeight, m.invertWidth, lod) : 0.0;
        float hY = w.y > 0.0 ? rmHeightSampleLod(uvY, layer, m.invertHeight, m.invertWidth, lod) : 0.0;
        float hZ = w.z > 0.0 ? rmHeightSampleLod(uvZ, layer, m.invertHeight, m.invertWidth, lod) : 0.0;
        return clamp(hX * w.x + hY * w.y + hZ * w.z, 0.0, 1.0);
    }
    return rmHeightSampleLod(dominantMappingUV(w, uvX, uvY, uvZ), layer,
                             m.invertHeight, m.invertWidth, lod);
}

// ── Secondary SDF scale patterns ──────────────────────────────────────────
// The pattern is evaluated in a cylindrical world frame whose axis follows the
// brush segment velocity. Triangular scales put their tip at -x (opposite the
// drag) and widen toward +x; hexagons use the same orientation. The stable
// frame is what keeps the scalar field continuous across octree cells.
// Everything is GPU-only: no scale is stored in the octree.

float rmSdHexagon2(vec2 p, float r) {
    const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
    p = abs(p);
    p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
    p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

float rmTriangleScale(vec2 q) {
    // q is centered in one cell. Tip at x=-0.5, base at x=+0.5.
    float x = q.x + 0.5;                 // 0 at tip, 1 at base
    float halfWidth = 0.55 * x;
    float edge = halfWidth - abs(q.y);
    float shape = smoothstep(0.07, 0.0, edge);
    shape *= smoothstep(1.0, 0.90, x);   // bevel the trailing base
    shape *= smoothstep(0.0, 0.06, x);
    return shape;
}

float rmHexagonScale(vec2 q) {
    q.y *= 1.15;
    float d = rmSdHexagon2(q, 0.34);
    float shape = smoothstep(0.055, -0.005, d);
    // The -x corner is the leading point; make it the high edge and taper
    // toward the trailing +x side.
    shape *= mix(0.55, 1.0, smoothstep(0.45, -0.30, q.x));
    return shape;
}

// ── Platonic-solid secondary patterns ─────────────────────────────────────
// Convex polyhedra are represented by their normalized face planes. The max
// plane distance is a valid defining field for the solid (and is exact along
// each face normal); it is enough for a displacement relief and keeps the
// five solids cheap enough for a per-sample pattern.
const vec3 RM_TETRA_N[4] = vec3[4](
    vec3( 0.577350269,  0.577350269,  0.577350269),
    vec3( 0.577350269, -0.577350269, -0.577350269),
    vec3(-0.577350269,  0.577350269, -0.577350269),
    vec3(-0.577350269, -0.577350269,  0.577350269));
const vec3 RM_DODECA_N[12] = vec3[12](
    vec3(0.0, 0.525731112, 0.850650808), vec3(0.0, 0.525731112, -0.850650808),
    vec3(0.0, -0.525731112, 0.850650808), vec3(0.0, -0.525731112, -0.850650808),
    vec3(0.525731112, 0.850650808, 0.0), vec3(0.525731112, -0.850650808, 0.0),
    vec3(-0.525731112, 0.850650808, 0.0), vec3(-0.525731112, -0.850650808, 0.0),
    vec3(0.850650808, 0.0, 0.525731112), vec3(0.850650808, 0.0, -0.525731112),
    vec3(-0.850650808, 0.0, 0.525731112), vec3(-0.850650808, 0.0, -0.525731112));
const vec3 RM_ICOSA_N[20] = vec3[20](
    vec3( 0.577350269,  0.577350269,  0.577350269), vec3( 0.577350269,  0.577350269, -0.577350269),
    vec3( 0.577350269, -0.577350269,  0.577350269), vec3( 0.577350269, -0.577350269, -0.577350269),
    vec3(-0.577350269,  0.577350269,  0.577350269), vec3(-0.577350269,  0.577350269, -0.577350269),
    vec3(-0.577350269, -0.577350269,  0.577350269), vec3(-0.577350269, -0.577350269, -0.577350269),
    vec3(0.0, 0.525731112, 0.850650808), vec3(0.0, 0.525731112, -0.850650808),
    vec3(0.0, -0.525731112, 0.850650808), vec3(0.0, -0.525731112, -0.850650808),
    vec3(0.525731112, 0.850650808, 0.0), vec3(0.525731112, -0.850650808, 0.0),
    vec3(-0.525731112, 0.850650808, 0.0), vec3(-0.525731112, -0.850650808, 0.0),
    vec3(0.850650808, 0.0, 0.525731112), vec3(0.850650808, 0.0, -0.525731112),
    vec3(-0.850650808, 0.0, 0.525731112), vec3(-0.850650808, 0.0, -0.525731112));

float rmPlatonicSdf(int type, vec3 p) {
    if (type == 5) { // tetrahedron
        float d = dot(p, RM_TETRA_N[0]);
        for (int i = 1; i < 4; ++i) d = max(d, dot(p, RM_TETRA_N[i]));
        return d - 0.14;
    }
    if (type == 6) { // cube
        return max(max(abs(p.x), abs(p.y)), abs(p.z)) - 0.27;
    }
    if (type == 7) { // octahedron
        return (abs(p.x) + abs(p.y) + abs(p.z)) * 0.577350269 - 0.20;
    }
    if (type == 8) { // dodecahedron
        float d = dot(p, RM_DODECA_N[0]);
        for (int i = 1; i < 12; ++i) d = max(d, dot(p, RM_DODECA_N[i]));
        return d - 0.20;
    }
    // icosahedron
    float d = dot(p, RM_ICOSA_N[0]);
    for (int i = 1; i < 20; ++i) d = max(d, dot(p, RM_ICOSA_N[i]));
    return d - 0.18;
}

float rmPlatonicPatternDisplacement(RmDisplacement d, vec3 p) {
    float repeat = max(d.patternRepeat, 1.0e-4);
    float intensity = clamp(d.patternIntensity, 0.0, 1.0);
    float age = max(ubo.brushPhase - d.patternTime, 0.0);

    vec3 axis = d.patternVelocity;
    if (dot(axis, axis) < 1.0e-8) axis = vec3(0.0, 1.0, 0.0);
    axis = normalize(axis);
    vec3 helper = abs(axis.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = normalize(cross(helper, axis));
    vec3 normal = normalize(cross(axis, bitangent));

    // The lattice is advected with the full 3D velocity; intensity also
    // controls how fast the solids spin.
    vec3 q = p / repeat + d.patternVelocity * (age * (0.25 + 0.75 * intensity) / repeat)
             + vec3(age * 0.025);
    vec3 cell = floor(q);
    vec3 f = fract(q) - 0.5;

    // Orientation: local +Y follows velocity, then each cell spins around that
    // axis with a cell-specific phase so the surface has a coherent but varied
    // rotation field.
    mat3 frame = mat3(bitangent, axis, normal);
    vec3 local = transpose(frame) * f;
    float phase = float(sdfHashU(uvec3(ivec3(cell)))) * (6.28318530718 / 4294967295.0);
    float spin = (age * (0.35 + 0.65 * intensity) + phase);
    float cs = cos(spin), sn = sin(spin);
    local.xz = mat2(cs, -sn, sn, cs) * local.xz;

    float sd = rmPlatonicSdf(d.pattern, local);
    float shape = smoothstep(0.055, 0.0, sd);
    return shape * d.patternDisplacement * d.strength * d.patternIntensity;
}

float rmVoronoi2D(vec2 uv) {
    vec2 cell = floor(uv);
    vec2 f = fract(uv);
    float minDist = 8.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 g = vec2(float(i), float(j));
            vec2 lattice = cell + g;
            uint h0 = sdfHashU(uvec3(ivec3(lattice, 0)));
            uint h1 = sdfHashU(uvec3(ivec3(lattice, 1)));
            vec2 seed = vec2(float(h0) * (1.0 / 4294967295.0),
                             float(h1) * (1.0 / 4294967295.0));
            minDist = min(minDist, length(g + seed - f));
        }
    }
    return 1.0 - smoothstep(0.0, 0.72, minDist);
}

float rmNoisePatternDisplacement(RmDisplacement d, vec3 p) {
    float repeat = max(d.patternRepeat, 1.0e-4);
    float age = max(ubo.brushPhase - d.patternTime, 0.0);
    float intensity = clamp(d.patternIntensity, 0.0, 1.0);
    float motion = age * (0.25 + 0.75 * intensity);

    if (d.pattern == 3) {
        // 3D Perlin: the domain is advected by the full velocity vector, so
        // the noise travels along the brush trail as it animates.
        vec3 q = p / repeat + d.patternVelocity * (motion / repeat)
                 + vec3(age * 0.035);
        float n = fbm(q, 4, 0.5, 2.0);
        // Boost the low-contrast normalized FBM into a full 0..1 relief.
        return smoothstep(-0.35, 0.35, n);
    }

    // Voronoi: a 2D cellular field in the same cylindrical frame as the scale
    // patterns. This keeps the 9-cell search cheap while still advecting with
    // the 3D velocity and time.
    vec3 axis = d.patternVelocity;
    if (dot(axis, axis) < 1.0e-8) axis = vec3(0.0, 1.0, 0.0);
    axis = normalize(axis);
    vec3 helper = abs(axis.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = normalize(cross(helper, axis));
    vec3 normal = normalize(cross(axis, bitangent));
    vec3 radial = p - axis * dot(p, axis);
    float radius = max(length(radial), repeat * 0.25);
    float angle = atan(dot(radial, bitangent), dot(radial, normal));
    vec2 uv = vec2(dot(p, axis), angle * radius) / repeat;
    uv.x += motion * (0.25 + min(length(d.patternVelocity) / repeat, 2.0) * 0.5);
    uv.y += age * 0.05;
    return rmVoronoi2D(uv);
}

float rmScalePatternHeight(vec2 uv, int type) {
    float row = floor(uv.y);
    uv.x += 0.5 * mod(row, 2.0);
    vec2 q = fract(uv) - 0.5;
    return (type == 1) ? rmTriangleScale(q) : rmHexagonScale(q);
}

float rmPatternDisplacement(RmDisplacement d, vec3 p) {
    if (d.pattern >= 5) {
        return rmPlatonicPatternDisplacement(d, p);
    }
    if (d.pattern >= 3) {
        float shape = rmNoisePatternDisplacement(d, p);
        return shape * d.patternDisplacement * d.strength * d.patternIntensity;
    }

    // Cylindrical frame around the drag axis. Unlike a per-sample surface
    // tangent frame, this is continuous across octree cells (the scalar
    // pattern does not inherit trilinear normal jumps), which is essential for
    // a stable displaced zero set.
    vec3 axis = d.patternVelocity;
    if (dot(axis, axis) < 1.0e-8) axis = vec3(0.0, 1.0, 0.0);
    axis = normalize(axis);
    vec3 helper = abs(axis.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = normalize(cross(helper, axis));
    vec3 normal = normalize(cross(axis, bitangent));

    float repeat = max(d.patternRepeat, 1.0e-4);
    vec3 radial = p - axis * dot(p, axis);
    float radius = max(length(radial), repeat * 0.25);
    float angle = atan(dot(radial, bitangent), dot(radial, normal));
    vec2 uv = vec2(dot(p, axis), angle * radius) / repeat;

    float speed = length(d.patternVelocity);
    float age = max(ubo.brushPhase - d.patternTime, 0.0);
    float intensity = clamp(d.patternIntensity, 0.0, 1.0);
    uv.x += age * (0.20 + 0.35 * min(speed / repeat, 2.0)) *
            (0.25 + 0.75 * intensity);
    float shape = rmScalePatternHeight(uv, d.pattern);
    float pulse = 0.90 + 0.10 * sin(ubo.brushPhase * 2.0 + uv.y * 6.28318530718);
    return shape * d.patternDisplacement * d.strength * d.patternIntensity * pulse;
}

// Signed world-space displacement at `p`. `n` is the base SDF normal used to
// select the triplanar projection (the same normal the old bump used).
float rmDisplacementValue(RmDisplacement d, MaterialNamed m, int material,
                          int texType, vec2 uvScale, vec3 p, vec3 n, float lod) {
    if (!d.enabled) return 0.0;

    if (!d.legacy && d.pattern > 0) {
        float disp = rmPatternDisplacement(d, p);
        return clamp(disp, -d.bounds, d.bounds);
    }

    float h;
    if (d.legacy) {
        // Bit-identical to the previous tessellation bump path.
        if (mappingUsesTriplanar(texType)) {
            h = sampleHeightTriplanarWLod(p, n, computeTriplanarWeights(n),
                                          material, uvScale, lod);
        } else {
            vec2 uvX, uvY, uvZ;
            vec3 w;
            computeMaterialUVs(texType, material, p, n, uvScale,
                               ubo.triplanarThreshold, ubo.triplanarExponent,
                               uvX, uvY, uvZ, w);
            h = sampleHeightLod(dominantMappingUV(w, uvX, uvY, uvZ), material, lod);
        }
    } else {
        h = rmSampleDisplacementHeight(d, m, texType, uvScale, p, n, lod);
    }

    float norm = h - 0.5 + d.offset;
    float disp = norm * d.amplitude * d.strength;
    float bound = d.bounds > 0.0 ? d.bounds : abs(d.amplitude) * d.strength;
    return clamp(disp, -bound, bound);
}

// World-space feature wavelength used by the conservative step. For the new
// path this is the inverse of the effective UV scale (the material frequency
// is the declared band limit). For legacy it is the raster tessellation
// spacing the old march used.
float rmDisplacementWavelength(RmDisplacement d, vec2 uvScale,
                               float nodeSize, float legacyTessLevels) {
    if (!d.enabled) return 0.0;
    if (d.legacy) {
        return nodeSize / clamp(legacyTessLevels, 1.0, 64.0);
    }
    if (d.pattern > 0) return max(d.patternRepeat * 0.10, 1.0e-4);
    vec2 s = abs(uvScale) * d.frequency;
    float scale = max(max(s.x, s.y), 1e-6);
    return 1.0 / scale;
}

#endif // OCTREE_DISPLACEMENT_FIELD_GLSL
