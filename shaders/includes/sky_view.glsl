#ifndef SKY_VIEW_GLSL
#define SKY_VIEW_GLSL

// Named view over the packed SkyUBO (declared in ubo.glsl, which must be
// included first). It lives in its own include on purpose: a global
// initializer is a STATIC use of the descriptor, so keeping it out of
// ubo.glsl means only the stages that actually read the sky reference the
// binding. The vegetation/indirect path and the tessellation control stage,
// which never read it, would otherwise need the sky UBO bound and valid.
SkyParamsNamed sky = skyParamsNamed();

#endif // SKY_VIEW_GLSL
