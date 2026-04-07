#version 450

// Converts a cubemap texture to an equirectangular 2D projection.
// Samples from a cubemap (set 0, binding 0) using direction vectors
// derived from the equirect UV convention matching water.frag / postprocess.frag:
//   uv.x = atan(z, x) / (2*PI) + 0.5
//   uv.y = acos(y) / PI

layout(set = 0, binding = 0) uniform samplerCube cubemapTex;

layout(push_constant) uniform PushConstants {
    vec2 resolution; // width, height of equirect target
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

void main() {
    // Normalize fragment coords to [0,1].
    vec2 uv = gl_FragCoord.xy / pc.resolution;

    // theta: polar angle from zenith (0 at top, PI at bottom)
    // phi: azimuthal angle (-PI to PI)
    float theta = uv.y * PI;
    float phi   = (uv.x - 0.5) * 2.0 * PI;

    // Spherical to Cartesian (Y-up), matching sky_equirect.frag convention
    vec3 dir;
    dir.x = sin(theta) * cos(phi);
    dir.y = cos(theta);
    dir.z = sin(theta) * sin(phi);

    // Correct orientation for specific cubemap faces by mirroring
    float ax = abs(dir.x);
    float ay = abs(dir.y);
    float az = abs(dir.z);
    // +Y face: mirror horizontally by flipping X
    if (ay >= ax && ay >= az && dir.y > 0.0) {
        dir.x = -dir.x;
    }
    // -X face: mirror horizontally by flipping Z
    else if (ax >= ay && ax >= az && dir.x < 0.0) {
        dir.z = -dir.z;
    }
    // +Z face: mirror horizontally by flipping X
    else if (az >= ax && az >= ay && dir.z > 0.0) {
        dir.x = -dir.x;
    }
    // -Z face: mirror horizontally by flipping X
    else if (az >= ax && az >= ay && dir.z < 0.0) {
        dir.x = -dir.x;
    }
    outColor = texture(cubemapTex, dir);
}
