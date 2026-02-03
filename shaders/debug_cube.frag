#version 450

layout(location = 0) in vec3 fragTexCoord;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 1) uniform sampler2D gridTexture;

void main() {
    // Sample the grid texture using XY coordinates (triplanar could be used too)
    vec4 texColor = texture(gridTexture, fragTexCoord.xy);
    
    // Modulate grid texture by node color to tint it while keeping pattern
    vec3 finalColor = texColor.rgb * fragColor;
    outColor = vec4(finalColor, 1.0);
}
