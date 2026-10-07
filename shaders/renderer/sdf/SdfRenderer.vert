#version 450

#include "../../types/SdfContainer.glsl"

// Generic SDF proxy vertex shader: one unit-cube instance per CONTAINER.
// The proxy bounds per-pixel work to visible containers; the fragment shader
// traverses the container's uniform grid (definitions/materials/instances).
// Mesh: unit cube positions in [0,1]^3. worldPos = mix(min,max,inPosition).

#include "../../includes/Locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;

layout(location = VARY_POSWORLD) out vec3 fragWorldPos;
layout(location = VARY_BRUSHPATCH) flat out int fragContainerIndex;

#include "../../includes/SceneBindings.glsl"
#include "../../includes/sdf/SdfMaterial.glsl"

layout(std430, set = 1, binding = 3) readonly buffer SdfContainerBuffer {
    SdfContainer sdfContainers[];
};

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
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
}
