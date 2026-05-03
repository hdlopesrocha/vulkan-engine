#version 450
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in vec3 fragTexCoordIn[];
layout(location = 1) flat in int fragTexIndexIn[];
layout(location = 2) in vec3 fragWorldPosIn[];

layout(location = 0) out vec3 inTexCoord;
layout(location = 1) flat out int inTexIndex;

// Must match SolidParamsUBO — only read the first two fields.
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
};

void main() {
    // Sentinel: w=-1 means this instance was skipped by the compute shader (slope filter).
    // Discard rather than rendering garbage geometry from uninitialized memory.
    if (fragTexIndexIn[0] < 0) return;

    vec3 worldPos = fragWorldPosIn[0];
    vec3 camPos   = ubo.viewPos.xyz;

    // Build camera-facing billboard axes.
    vec3 toCamera = normalize(camPos - worldPos);
    vec3 worldUp  = vec3(0.0, 1.0, 0.0);
    vec3 right    = normalize(cross(worldUp, toCamera));
    vec3 up       = normalize(cross(toCamera, right));

    float hs = billboardScale * 0.5;
    int   ti = fragTexIndexIn[0];
    float layer = fragTexCoordIn[0].z;

    // Triangle strip: BL → BR → TL → TR
    gl_Position = ubo.viewProjection * vec4(worldPos - right * hs,                     1.0); inTexCoord = vec3(0.0, 1.0, layer); inTexIndex = ti; EmitVertex();
    gl_Position = ubo.viewProjection * vec4(worldPos + right * hs,                     1.0); inTexCoord = vec3(1.0, 1.0, layer); inTexIndex = ti; EmitVertex();
    gl_Position = ubo.viewProjection * vec4(worldPos - right * hs + up * billboardScale, 1.0); inTexCoord = vec3(0.0, 0.0, layer); inTexIndex = ti; EmitVertex();
    gl_Position = ubo.viewProjection * vec4(worldPos + right * hs + up * billboardScale, 1.0); inTexCoord = vec3(1.0, 0.0, layer); inTexIndex = ti; EmitVertex();
    EndPrimitive();
}
