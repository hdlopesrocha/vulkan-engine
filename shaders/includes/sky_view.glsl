#ifndef SKY_VIEW_GLSL
#define SKY_VIEW_GLSL

// The sky/cloud block (set=0 binding=6) is declared by ubo.glsl as the shared
// canonical `SkyUniform sky`. This include only pulls the type for stages
// that read it; include order still requires ubo.glsl first.

#include "../ubo/SkyUniform.glsl"

#endif // SKY_VIEW_GLSL
