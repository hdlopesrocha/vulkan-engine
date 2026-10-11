#version 450

// Grass-shadow proxy vertex shader: one unit-cube instance per GRASS SDF
// container (mirrors SdfRenderer.vert's container mapping: [0,1]^3 ->
// sdfContainers[gl_InstanceIndex].boundsMin/Max). The proxy is positioned
// with the cascade light view-projection from the push constants, because
// this pipeline's layout carries the SDF set at index 1 + push constants
// only; set 0 is the shadow pass's already-bound solid scene set and is
// neither bound nor read here. SceneBindings.glsl is included for parity
// with SdfRenderer.vert (its declarations are statically unused and are
// eliminated by the driver).

#include "../../types/SdfContainer.glsl"

#include "../../includes/Locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;

layout(location = VARY_POSWORLD) out vec3 fragWorldPos;
layout(location = VARY_BRUSHPATCH) flat out int fragContainerIndex;
// M10 (report 27): march scalars derived once per vertex (identical ops to
// the old per-fragment derivation in SdfGrassShadow.frag, deleted there);
// flat varyings carry them unchanged to every fragment of the proxy.
// Locations 16/17 are pipeline-local (no geometry stage here); Locations.glsl
// is untouched.
layout(location = 16) flat out vec4 vShadowMarch; // x=eps, y=safety, z=minStep, w=maxStep
layout(location = 17) flat out int vShadowMaxSteps;

#include "../../includes/SceneBindings.glsl"
#include "../../includes/sdf/SdfMaterial.glsl"

layout(std430, set = 1, binding = 3) readonly buffer SdfContainerBuffer {
    SdfContainer sdfContainers[];
};

// Push constants (128 B; C++ twin SdfGrassShadowPC in SdfRenderer.hpp).
// The impostor member is statically unused in the vertex stage but keeps the
// block layout twin with the fragment stage and the C++ struct (the old block
// omitted it, shifting lightDir to offset 96 vs 112 in the C++ struct).
layout(push_constant) uniform SdfGrassShadowPC {
    mat4 lightViewProj; // cascade light view-projection (world -> light clip)
    vec4 params;        // x = time (s), y = max steps, z = epsilon, w = safety
    vec4 march;         // x = max step (m), y = min step (m), z = shadow LOD camScale
    vec4 impostor;      // x = impostor fade start, y = fade full (unused in VS)
    vec4 lightDir;      // xyz = light-to-scene direction (world), w unused
} pc;

void main() {
    uint idx = uint(gl_InstanceIndex);
    vec3 bMin = vec3(0.0);
    vec3 bMax = vec3(1.0);
    if (idx < uint(sdfContainers.length())) {
        bMin = sdfContainers[idx].boundsMin;
        bMax = sdfContainers[idx].boundsMax;
    }
    vec3 f = clamp(inPosition, vec3(0.0), vec3(1.0));
    vec3 worldPos = mix(bMin, bMax, f);
    fragWorldPos = worldPos;
    fragContainerIndex = int(idx);
    gl_Position = pc.lightViewProj * vec4(worldPos, 1.0);
    // M10 (report 27): cascade march constants, verbatim from the deleted
    // per-fragment derivation (same operations in the same order, evaluated
    // once per vertex instead of once per fragment). The 256 clamp mirrors
    // SDF_GRASS_SHADOW_MAX_STEPS_HARD (proof at SdfGrassShadow.frag:67-68).
    int vSteps = int(pc.params.y + 0.5);
    if (vSteps <= 0) vSteps = 64;
    vSteps = min(vSteps, 256);
    vShadowMaxSteps = vSteps;
    float vEps = (pc.params.z > 0.0) ? pc.params.z : 0.01;
    float vSafety = clamp((pc.params.w > 0.0) ? pc.params.w : 0.7, 0.1, 1.0);
    float vMinStep = max(pc.march.y, 1e-4);
    float vMaxStep = max((pc.march.x > 0.0) ? pc.march.x : 1.0, vMinStep);
    vShadowMarch = vec4(vEps, vSafety, vMinStep, vMaxStep);
}
