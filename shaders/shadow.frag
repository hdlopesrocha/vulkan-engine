#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) in vec3 fragPosWorld;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=unused, w=flipParallaxDirection
} push;

layout(binding = 3) uniform sampler2DArray heightArray;

void main() {
    // Shadow map should NOT apply parallax displacement
    // We render the actual geometry positions without displacement
    // This prevents lateral misalignment of faces in the shadow map
    // Parallax is only applied in the main render pass for visual detail
    
    // Depth is automatically written by the depth attachment
}
