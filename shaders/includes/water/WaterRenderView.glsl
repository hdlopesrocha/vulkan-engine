#ifndef WATER_RENDER_VIEW_GLSL
#define WATER_RENDER_VIEW_GLSL

// The water render block (set=0 binding=10) is declared by ubo.glsl as the
// shared canonical WaterRenderUBO. This include only pulls the type for the
// water stages that read it; include order still requires ubo.glsl first.
#include "../../ubo/WaterRenderUBO.glsl"

#endif // WATER_RENDER_VIEW_GLSL
