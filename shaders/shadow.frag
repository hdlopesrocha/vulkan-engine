#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 3) in vec3 fragPosWorld;

#include "includes/ubo.glsl"

layout(binding = 3) uniform sampler2DArray heightArray;

void main() {
    // Depth is automatically written by the depth attachment
}
