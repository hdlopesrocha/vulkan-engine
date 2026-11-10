#ifndef SDF_OPERATION_EULER_MAT_GLSL
#define SDF_OPERATION_EULER_MAT_GLSL

// Rotation from XYZ euler angles (radians). Order: R = Rx * Ry * Rz, i.e.
// intrinsic XYZ. Shared by the generic SdfModel (SdfModel.glsl) so every GPU
// SDF uses one rotation convention.
mat3 sdfEulerMat(vec3 euler) {
    float cx = cos(euler.x);
    float sx = sin(euler.x);
    float cy = cos(euler.y);
    float sy = sin(euler.y);
    float cz = cos(euler.z);
    float sz = sin(euler.z);
    mat3 rx = mat3(1.0, 0.0, 0.0,
                  0.0, cx, sx,
                  0.0, -sx, cx);
    mat3 ry = mat3(cy, 0.0, -sy,
                  0.0, 1.0, 0.0,
                  sy, 0.0, cy);
    mat3 rz = mat3(cz, sz, 0.0,
                  -sz, cz, 0.0,
                  0.0, 0.0, 1.0);
    return rx * ry * rz;
}

#endif // SDF_OPERATION_EULER_MAT_GLSL
