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
    if (material < 0) return d;

    MaterialNamed m = materialNamed(materials[material]);

    // The new material path takes precedence when configured; otherwise keep
    // the legacy tessellation bump bit-compatible.
    const bool newConfigured = ubo.displacementEnabled != 0u &&
        m.displacementStrength > 0.0 && m.displacementAmplitude != 0.0;
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

    d.enabled = true;
    d.amplitude = m.displacementAmplitude;
    d.frequency = max(m.displacementFrequency, 1e-4);
    d.offset = m.displacementOffset;
    d.strength = clamp(m.displacementStrength, 0.0, 1.0);
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

// Signed world-space displacement at `p`. `n` is the base SDF normal used to
// select the triplanar projection (the same normal the old bump used).
float rmDisplacementValue(RmDisplacement d, MaterialNamed m, int material,
                          int texType, vec2 uvScale, vec3 p, vec3 n, float lod) {
    if (!d.enabled) return 0.0;

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
    vec2 s = abs(uvScale) * d.frequency;
    float scale = max(max(s.x, s.y), 1e-6);
    return 1.0 / scale;
}

#endif // OCTREE_DISPLACEMENT_FIELD_GLSL
