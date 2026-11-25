// Canonical debug view IDs — mirror of vulkan/includes/debug/DebugModes.hpp.
// Keep the enum, names and helper in sync with the C++ header.
//
// The solid (SolidRenderer.frag) and water (water*.frag) fragment paths dispatch
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
const int DEBUG_MODE_TESS_HEAT = 18;
const int DEBUG_MODE_SKY_REFLECTION = 19;
const int DEBUG_MODE_REFRACTION_COLOR = 20;
const int DEBUG_MODE_WATER_NOISE = 21;
const int DEBUG_MODE_DISPLACEMENT = 22;
const int DEBUG_MODE_THICKNESS = 23;
const int DEBUG_MODE_ABSORPTION = 24;
const int DEBUG_MODE_CAUSTICS = 25;
const int DEBUG_MODE_DEPTH_SOURCE = 26;
const int DEBUG_MODE_WATER_COMPOSE = 27;
const int DEBUG_MODE_WATER_COLOR = 28;
const int DEBUG_MODE_WATER_DEPTH_SOURCES = 29;
const int DEBUG_MODE_SCENE_DEPTH = 30;
const int DEBUG_MODE_REFLECTION_SOURCE = 31;
const int DEBUG_MODE_WATER_SLOPE = 32;
const int DEBUG_MODE_WATER_CONTACT = 33;
const int DEBUG_MODE_WATER_SWELL = 34;
const int DEBUG_MODE_WATER_FOAM_MASK = 35;
const int DEBUG_MODE_WATER_SPECULAR_NOISE = 36;
const int DEBUG_MODE_SHORE_DIRECTION = 37;
const int DEBUG_MODE_WATER_GERSTNER = 38;
const int DEBUG_MODE_VEGETATION_LOD = 39;
const int DEBUG_MODE_CLOUD_SDF = 40;
const int DEBUG_MODE_CLOUD_BASE_DENSITY = 41;
const int DEBUG_MODE_CLOUD_COVERAGE = 42;
const int DEBUG_MODE_CLOUD_WARP = 43;
const int DEBUG_MODE_CLOUD_EROSION = 44;
const int DEBUG_MODE_CLOUD_FINAL_DENSITY = 45;
const int DEBUG_MODE_CLOUD_LIGHTING = 46;
const int DEBUG_MODE_CLOUD_SHADOWS = 47;
const int DEBUG_MODE_CLOUD_STEPS = 48;
const int DEBUG_MODE_CLOUD_SKIP = 49;
const int DEBUG_MODE_OCTREE_SDF_RAYMARCH = 50;
// True raymarched displacement diagnostics (octree raymarch only).
const int DEBUG_MODE_ORIGINAL_SDF_SURFACE = 51;
const int DEBUG_MODE_DISPLACEMENT_MAGNITUDE = 52;
const int DEBUG_MODE_DISPLACEMENT_BOUNDS = 53;
const int DEBUG_MODE_DISPLACED_NORMALS = 54;
const int DEBUG_MODE_DISPLACEMENT_ITERATIONS = 55;
const int DEBUG_MODE_DISPLACEMENT_STEP = 56;
const int DEBUG_MODE_DISPLACEMENT_RESIDUAL = 57;
const int DEBUG_MODE_DISPLACEMENT_SHADOW = 58;
