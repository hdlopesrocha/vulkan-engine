// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    // Material flags.
    // x=unused, y=unused, z=ambient, w=unused
    vec4 materialFlags;
    vec4 mappingParams; // x=mappingEnabled (0=off,1=on) toggles tessellation + bump mapping, y/z/w unused
    vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    vec4 triplanarParams; // x=scaleU, y=scaleV, z=enabled(1.0), w=unused
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x/y/z = unused, w=global shadows enabled (1.0 = on)
    vec4 debugParams; // x=debugMode (0=normal,1=fragment normal,2=normal map (world),3=uv,4=tangent,5=bitangent,6=geometry normal (world),7=albedo,8=normal texture,9=height/bump,10=lighting (N·L,shadow),11=normal from derivatives,12=light vector (rgb),13=N·L grayscale,14=shadow diagnostics)
} ubo;