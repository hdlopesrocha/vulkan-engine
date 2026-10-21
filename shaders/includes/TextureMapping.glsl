#ifndef TEXTURE_MAPPING_GLSL
#define TEXTURE_MAPPING_GLSL

// Texture-coordinate mapping shared by the solid shading and displacement
// paths. The technique is a per-vertex value (Vertex::textureType, painted by
// the TexturePainter) that mirrors TextureType in math/TextureType.hpp:
//   0 = triplanar, 1/2/3 = Projection X/Y/Z, 4 = Angular, 5 = Custom (UV).
//
// Requires the caller to have declared materials[] / materialNamed()
// (SceneBindings.glsl) before including this file.

#include "../types/MaterialNamed.glsl"

// Mapping modes (keep in sync with TextureType in math/TextureType.hpp).
const int TEXTURE_MAPPING_TRIPLANAR = 0;
const int TEXTURE_MAPPING_PROJECTION_X = 1;
const int TEXTURE_MAPPING_PROJECTION_Y = 2;
const int TEXTURE_MAPPING_PROJECTION_Z = 3;
const int TEXTURE_MAPPING_ANGULAR = 4;
const int TEXTURE_MAPPING_CUSTOM = 5;

// Projection axis accepted by textureMappingUV().
const int TEXTURE_AXIS_X = 0;
const int TEXTURE_AXIS_Y = 1;
const int TEXTURE_AXIS_Z = 2;

// True when the material uses the weighted triplanar path. Mode 0 with the
// legacy triplanar flag cleared is a UV-mapped material and behaves as Custom.
bool materialUsesTriplanarMapping(in MaterialNamed m) {
    return m.textureMapping == TEXTURE_MAPPING_TRIPLANAR && m.triplanarEnabled;
}

// Same test for an explicit per-vertex mode (Vertex::textureType).
bool mappingUsesTriplanar(in int mode, in MaterialNamed m) {
    return mode == TEXTURE_MAPPING_TRIPLANAR && m.triplanarEnabled;
}

// World-space projection UV for one axis. Same conventions as the legacy
// triplanar path (Triplanar.glsl): the geometric-normal sign selects the
// mirrored side so both sides of a surface read the texture right-side up,
// and triplanarScaleU/V scale the result.
vec2 textureMappingUV(in MaterialNamed m, in int axis, in vec3 pos, in vec3 geomN) {
    vec2 scale = vec2(m.triplanarScaleU, m.triplanarScaleV);
    vec2 uv = vec2(0.0);
    if (axis == TEXTURE_AXIS_X) {
        uv = (geomN.x >= 0.0) ? vec2(-pos.z, -pos.y) : vec2(pos.z, -pos.y);
    } else if (axis == TEXTURE_AXIS_Y) {
        uv = (geomN.y >= 0.0) ? vec2(pos.x, pos.z) : vec2(pos.x, -pos.z);
    } else {
        uv = (geomN.z >= 0.0) ? vec2(pos.x, -pos.y) : vec2(-pos.x, -pos.y);
    }
    return uv * scale;
}

// Angular mode: spherical coordinates around the material's mapping origin. The
// volume radius (mappingOrigin.w) sets the tiling; a zero radius falls back to
// a unit sphere so the UV never degenerates.
vec2 textureMappingAngularUV(in MaterialNamed m, in vec3 pos) {
    vec3 d = pos - m.mappingOrigin.xyz;
    float r = max(length(d), 1e-5);
    vec2 uv = vec2(0.5 + atan(d.z, d.x) / 6.28318530718,
                   acos(clamp(d.y / r, -1.0, 1.0)) / 3.14159265359);
    return uv * max(m.mappingOrigin.w, 1.0) * 0.01;
}

// Triplanar blend weights from the geometric normal, using the caller's
// dead-zone threshold and exponent. Same math as computeTriplanarWeights()
// in Common.glsl, parameterized so this include does not depend on the UBO.
vec3 computeMappingTriplanarWeights(in vec3 geomN, in float threshold, in float exponent) {
    vec3 w = abs(normalize(geomN));
    vec3 wt = max(vec3(0.0), w - vec3(threshold));
    float e = max(1.0, exponent);
    if (e > 1.0) wt = pow(wt, vec3(e));
    float sum = wt.x + wt.y + wt.z + 1e-6;
    return wt / sum;
}

// UV of the largest-weight projection (ties: X, then Y, matching the existing
// dominant-projection helpers). One-hot weights return their own slot.
vec2 dominantMappingUV(in vec3 weights, in vec2 uvX, in vec2 uvY, in vec2 uvZ) {
    float dx = step(weights.y, weights.x) * step(weights.z, weights.x);
    float dy = (1.0 - dx) * step(weights.z, weights.y);
    return uvX * dx + uvY * dy + uvZ * (1.0 - dx - dy);
}

// Per-corner UVs and blend weights for one vertex's texture-coordinate mode.
//   mode        - per-vertex TextureType (Vertex::textureType)
//   uvX/uvY/uvZ - projection UVs; unused slots are zero.
//   weights     - one-hot for single projections / Angular / Custom, and the
//                 geometric-normal triplanar weights for mode 0.
// threshold/exponent are ubo.triplanarThreshold / ubo.triplanarExponent.
void computeMaterialUVs(in int mode, in int brushIndex, in vec3 pos, in vec3 geomN, in vec2 fragUV,
                        in float threshold, in float exponent,
                        out vec2 uvX, out vec2 uvY, out vec2 uvZ, out vec3 weights) {
    MaterialNamed m = materialNamed(materials[brushIndex]);
    uvX = vec2(0.0);
    uvY = vec2(0.0);
    uvZ = vec2(0.0);
    weights = vec3(1.0, 0.0, 0.0);

    if (mode == TEXTURE_MAPPING_PROJECTION_X) {
        uvX = textureMappingUV(m, TEXTURE_AXIS_X, pos, geomN);
    } else if (mode == TEXTURE_MAPPING_PROJECTION_Y) {
        uvY = textureMappingUV(m, TEXTURE_AXIS_Y, pos, geomN);
        weights = vec3(0.0, 1.0, 0.0);
    } else if (mode == TEXTURE_MAPPING_PROJECTION_Z) {
        uvZ = textureMappingUV(m, TEXTURE_AXIS_Z, pos, geomN);
        weights = vec3(0.0, 0.0, 1.0);
    } else if (mode == TEXTURE_MAPPING_ANGULAR) {
        uvX = textureMappingAngularUV(m, pos);
    } else if (mode == TEXTURE_MAPPING_CUSTOM) {
        uvX = fragUV;
    } else if (m.triplanarEnabled) {
        // Triplanar (mode 0); unknown modes fall back to the same projection.
        uvX = textureMappingUV(m, TEXTURE_AXIS_X, pos, geomN);
        uvY = textureMappingUV(m, TEXTURE_AXIS_Y, pos, geomN);
        uvZ = textureMappingUV(m, TEXTURE_AXIS_Z, pos, geomN);
        weights = computeMappingTriplanarWeights(geomN, threshold, exponent);
    } else {
        // Legacy mode 0 with triplanar disabled: behave as Custom (UV).
        uvX = fragUV;
    }
}

#endif // TEXTURE_MAPPING_GLSL
