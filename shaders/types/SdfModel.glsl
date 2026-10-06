#ifndef SDF_MODEL_GLSL
#define SDF_MODEL_GLSL

// Generic SDF model transform: every GPU SDF is evaluated in its own local
// frame and the model maps that frame to world space (translate * rotate *
// scale). Primitives (smoke, flame, sphere, box, ...) never transform
// points themselves; the evaluator asks the model instead.
// Helpers live in shaders/includes/sdf_model.glsl.
struct SdfModel {
    vec3 translation; // world-space position of the local origin
    vec3 rotation;    // XYZ euler radians, R = Rx * Ry * Rz (sdfEulerMat)
    vec3 scale;       // per-axis local -> world scale (positive)
};

#endif // SDF_MODEL_GLSL
