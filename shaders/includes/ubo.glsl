// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=flipParallaxDirection
    vec4 parallaxLOD; // x=parallaxNear, y=parallaxFar, z=reductionAtFar, w=unused
    vec4 mappingParams; // x=mappingMode (0=none,1=parallax,2=tessellation), y/z/w unused
    vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    vec4 triplanarParams; // x=scaleU, y=scaleV, z=enabled(1.0), w=unused
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x=enableSelfShadow, y=enableShadowDisplacement, z=selfShadowQuality, w=unused
} ubo;