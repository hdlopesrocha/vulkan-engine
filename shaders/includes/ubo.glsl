// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 mvp; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    // Material flags.
    // x=flipNormalY, y=flipTangentHandedness, z=ambient, w=unused
    vec4 materialFlags;
    vec4 mappingParams; // x=mappingEnabled (0=off,1=on) toggles tessellation + bump mapping, y/z/w unused
    vec4 specularParams; // x=specularStrength, y=shininess, z=unused, w=unused
    vec4 triplanarParams; // x=scaleU, y=scaleV, z=enabled(1.0), w=unused
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x/y/z = unused (self-shadow & displacement removed), w=global shadows enabled (1.0 = on)
    vec4 debugParams; // x=debugMode (0=normal,1=geometry normal,2=normal map,3=uv,4=tangent,5=bitangent,6=raw albedo,7=raw normal map,8=bump/height,9=TBN composite,10=pre-projection,11=normal from derivatives)
} ubo;