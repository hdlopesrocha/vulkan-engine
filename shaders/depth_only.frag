#version 450

#include "includes/locations.glsl"

layout(location = VARY_COLOR) in vec3 fragColor;
layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) in vec3 fragTexWeights;
layout(location = VARY_SHARPNORMAL) in vec3 fragSharpNormal;

void main() {}
