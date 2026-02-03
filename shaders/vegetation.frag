#version 450
layout(location = 0) in vec2 inTexCoord;
layout(location = 1) in flat int inTexIndex;
layout(location = 0) out vec4 outColor;


layout(set = 1, binding = 0) uniform sampler2DArray vegetationTextures;
layout(push_constant) uniform PushConstants {
    float billboardScale;
};

void main() {
    outColor = texture(vegetationTextures, vec3(inTexCoord, inTexIndex));
    if (outColor.a < 0.1) discard;
}
