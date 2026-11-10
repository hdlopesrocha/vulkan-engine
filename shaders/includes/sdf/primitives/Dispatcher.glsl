#ifndef SDF_PRIMITIVES_DISPATCH_GLSL
#define SDF_PRIMITIVES_DISPATCH_GLSL

// Dispatcher for the generic SDF primitive library. Each primitive lives in
// its own file under sdf/primitives/; include those directly if a shader only
// needs a subset.

#include "../../../types/SdfPrimitiveType.glsl"

#include "Sphere.glsl"
#include "Box.glsl"
#include "RoundedBox.glsl"
#include "Capsule.glsl"
#include "Cylinder.glsl"
#include "Cone.glsl"
#include "Torus.glsl"
#include "Plane.glsl"
#include "Flame.glsl"
#include "Smoke.glsl"

// Evaluates the primitive selected by primType. Unknown types return a large
// distance (treated as empty space, never as a surface).
float sdfPrimitive(vec3 p, uint primType, vec4 p0, vec4 p1) {
    if (primType == SDF_PRIM_SPHERE) {
        return sdSphere(p, p0, p1);
    }
    if (primType == SDF_PRIM_BOX) {
        return sdBox(p, p0, p1);
    }
    if (primType == SDF_PRIM_ROUNDED_BOX) {
        return sdRoundedBox(p, p0, p1);
    }
    if (primType == SDF_PRIM_CAPSULE) {
        return sdCapsule(p, p0, p1);
    }
    if (primType == SDF_PRIM_CYLINDER) {
        return sdCylinder(p, p0, p1);
    }
    if (primType == SDF_PRIM_CONE) {
        return sdCone(p, p0, p1);
    }
    if (primType == SDF_PRIM_TORUS) {
        return sdTorus(p, p0, p1);
    }
    if (primType == SDF_PRIM_PLANE) {
        return sdPlane(p, p0, p1);
    }
    if (primType == SDF_PRIM_FLAME) {
        return sdTaperedFlame(p, p0, p1);
    }
    if (primType == SDF_PRIM_SMOKE) {
        return sdSmoke(p, p0, p1);
    }
    return 1e5;
}

#endif // SDF_PRIMITIVES_DISPATCH_GLSL
