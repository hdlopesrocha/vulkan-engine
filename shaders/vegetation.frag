#version 450
layout(location = 0) in vec3 inTexCoord;  // xy=uv, z=array layer
layout(location = 1) in flat int inTexIndex;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2DArray albedoArray;
layout(set = 1, binding = 1) uniform sampler2DArray normalArray;
layout(set = 1, binding = 2) uniform sampler2DArray opacityArray;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float pad0;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

void main() {
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);
    outColor = texture(albedoArray, coord);
    float opacity = texture(opacityArray, coord).r;
    outColor.a *= opacity;
    if (outColor.a < 0.1) discard;
}
