#version 450

#include "includes/locations.glsl"

layout(location = VARY_UV) in vec3 fragTexCoord;
layout(location = VARY_COLOR) in vec3 fragColor;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

layout(set = 1, binding = 1) uniform sampler2D gridTexture;

void main() {
    // Sample the grid texture using XY coordinates (triplanar could be used too)
    vec4 texColor = texture(gridTexture, fragTexCoord.xy);
    
    // Modulate grid texture by node color to tint it while keeping pattern
    vec3 finalColor = texColor.rgb * fragColor;
    outColor = vec4(finalColor, 1.0);
}
