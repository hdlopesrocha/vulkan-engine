#version 450
// Geometry shader: forward per-vertex varyings from vertex shader to fragment shader
layout(points) in;
layout(points, max_vertices = 1) out;

layout(location = 0) in vec2 fragTexCoordIn[];
layout(location = 1) flat in int fragTexIndexIn[];

layout(location = 0) out vec2 inTexCoord;
layout(location = 1) flat out int inTexIndex;

layout(push_constant) uniform PushConstants {
    float billboardScale;
};

void main() {
    gl_Position = gl_in[0].gl_Position;
    // Forward varyings
    inTexCoord = fragTexCoordIn[0];
    inTexIndex = fragTexIndexIn[0];
    EmitVertex();
    EndPrimitive();
}
