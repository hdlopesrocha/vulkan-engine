#ifndef WATER_RENDER_VIEW_GLSL
#define WATER_RENDER_VIEW_GLSL

// Named view over the packed WaterRenderUBO (declared in ubo.glsl, which must
// be included first). Separate include for the same reason as sky_view.glsl:
// only the water stages that read the water time / lobe toggles statically
// reference the binding.
WaterRenderParamsNamed waterRenderUBO = waterRenderParamsNamed();

#endif // WATER_RENDER_VIEW_GLSL
