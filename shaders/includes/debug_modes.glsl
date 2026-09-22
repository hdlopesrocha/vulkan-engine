// Canonical debug view IDs — mirror of vulkan/includes/DebugModes.hpp.
// Keep the enum, names and helper in sync with the C++ header.
//
// The solid (main.frag) and water (main_water*.frag) fragment paths dispatch
// on the same IDs. Views that have no meaning on a surface fall through to
// that surface's normal shading, so both surfaces always honor the selection.
const int DEBUG_MODE_NONE = 0;
const int DEBUG_MODE_SHADING_NORMAL = 1;
const int DEBUG_MODE_GEOMETRIC_NORMAL = 2;
const int DEBUG_MODE_FACE_NORMAL = 3;
const int DEBUG_MODE_ALBEDO = 4;
const int DEBUG_MODE_NORMAL_MAP = 5;
const int DEBUG_MODE_HEIGHT_MAP = 6;
const int DEBUG_MODE_ROUGHNESS = 7;
const int DEBUG_MODE_AMBIENT_OCCLUSION = 8;
const int DEBUG_MODE_TRIPLANAR_WEIGHTS = 9;
const int DEBUG_MODE_MATERIAL_INDEX = 10;
const int DEBUG_MODE_UV = 11;
const int DEBUG_MODE_N_DOT_L = 12;
const int DEBUG_MODE_LIGHT_VECTOR = 13;
const int DEBUG_MODE_SHADOW = 14;
const int DEBUG_MODE_REFLECTION_COLOR = 15;
const int DEBUG_MODE_REFLECTION_VECTOR = 16;
const int DEBUG_MODE_FRESNEL = 17;
const int DEBUG_MODE_RAY_MASK = 18;
const int DEBUG_MODE_TESS_HEAT = 19;
const int DEBUG_MODE_SKY_REFLECTION = 20;
const int DEBUG_MODE_REFRACTION_COLOR = 21;
const int DEBUG_MODE_WATER_NOISE = 22;
const int DEBUG_MODE_DISPLACEMENT = 23;
const int DEBUG_MODE_THICKNESS = 24;
const int DEBUG_MODE_ABSORPTION = 25;
const int DEBUG_MODE_CAUSTICS = 26;
const int DEBUG_MODE_DEPTH_SOURCE = 27;
const int DEBUG_MODE_WATER_COMPOSE = 28;
const int DEBUG_MODE_WATER_REGIONS = 29;
const int DEBUG_MODE_WATER_DEPTH_SOURCES = 30;
const int DEBUG_MODE_SCENE_DEPTH = 31;
const int DEBUG_MODE_REFLECTION_SOURCE = 32;

// Views whose displayed value is produced by traced reflection/refraction
// rays force the full-quality reference path (full-rate + dual-trace) so the
// ray-budget cuts cannot distort them. Mirrors
// debugmode::debugModeForcesRtReference; ray mask / depth source stay
// budgeted because they visualize the live cuts themselves.
bool debugModeForcesRtReference(int mode) {
    return mode == DEBUG_MODE_REFLECTION_COLOR
        || mode == DEBUG_MODE_REFRACTION_COLOR
        || mode == DEBUG_MODE_THICKNESS
        || mode == DEBUG_MODE_REFLECTION_SOURCE
        || mode == DEBUG_MODE_SHADOW;
}
