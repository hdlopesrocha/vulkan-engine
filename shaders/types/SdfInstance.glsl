#ifndef SDF_INSTANCE_GLSL
#define SDF_INSTANCE_GLSL

// Canonical SDF instance (std430, 96 B). Single definition shared with the
// CPU scene builder: sdf/types/SdfInstance.hpp — identical layout, names
// and offsets.
// Offset table (verified against the C++ static_asserts):
//   position 0, scale 12, rotation 16, heightScale 28, radiusScale 32,
//   intensity 36, seed 40, defIdx 44, matIdx 48, containerIdx 52,
//   boundsMin 64, boundsMax 80.
// 56..64 / 76..80 / 92..96 are std430 padding (16-byte vector alignment and
// the struct-size multiple-of-16 rule); no explicit pad members are needed.
// position/rotation feed the generic SdfModel (shaders/types/SdfModel.glsl).
struct SdfInstance {
    vec3 position;    // offset  0  world position
    float scale;      // offset 12  uniform scale
    vec3 rotation;    // offset 16  euler radians, R = Rx*Ry*Rz (SdfModel)
    float heightScale;// offset 28
    float radiusScale;// offset 32
    float intensity;  // offset 36
    float seed;       // offset 40
    uint defIdx;      // offset 44
    uint matIdx;      // offset 48
    uint containerIdx;// offset 52
    vec3 boundsMin;   // offset 64  world AABB min
    vec3 boundsMax;   // offset 80  world AABB max
};

#endif // SDF_INSTANCE_GLSL
