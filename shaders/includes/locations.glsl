#ifndef LOCATIONS_GLSL
#define LOCATIONS_GLSL

// Vertex attribute locations (match VkVertexInputAttributeDescription in C++ pipelines)
#define ATTR_POS 0
#define ATTR_COLOR 1
#define ATTR_UV 2
#define ATTR_NORMAL 3
#define ATTR_BRUSH_INDEX 4
#define ATTR_HSV 6
#define ATTR_INSTANCE 5
// Baked per-instance vegetation height scale (perf report 22 C2/H4): float
// per instance in the concatenated aux buffer (vertex binding 2).
#define ATTR_VEG_AUX 7

// Fragment outputs (color attachments)
// Keep these matching render pass attachment locations (0 = primary color)
#define FRAG_OUT_COLOR 0
#define FRAG_OUT_NORMAL 1
#define FRAG_OUT_DEPTH 2
// Water aux attachments (color attachments 1 and 2 of the water geometry
// pass): body = refraction+tint color + weight, column = measured depth (m).
#define FRAG_OUT_WATER_BODY 1
#define FRAG_OUT_WATER_COLUMN 2

// Inter-stage varying locations.
// Important: geometry stage input interfaces are limited to 64 components
// (16 locations × 4 components). Keep any varyings used by geometry
// shaders at locations <= 15. We remap a subset used by geometry to a
// low contiguous range and place less-critical varyings above that.
#define VARY_SHARPNORMAL 6
#define VARY_UV 7
#define VARY_NORMAL 8
#define VARY_POSWORLD 9
#define VARY_BRUSHPATCH 10
#define VARY_POSLIGHT 11
#define VARY_PLANE_NORMAL 12
#define VARY_WATERDEPTH 12 // water: TES-measured water thickness (-1 = unknown/deep)
#define VARY_SHOREDIR 13   // water: unit shore direction from the depth gradient (toward thinner water)
#define VARY_FACE_NORMAL 13
#define VARY_ROTFRAC 14
#define VARY_LOCALNORMAL 15
#define VARY_TEXWEIGHTS 16
#define VARY_POSCLIP 17
#define VARY_DEBUG 18
#define VARY_TANGENTWS 19
#define VARY_LOCALPOS 20
#define VARY_SDF 21
#define VARY_BASEPOS 21   // water: undisplaced base world position + raw bump amplitude (xyz=pos, w=amp)
#define VARY_COLOR 22
#define VARY_HSV 23

#endif // LOCATIONS_GLSL
