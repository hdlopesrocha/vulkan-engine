// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 viewProjection; mat4 model; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 viewProjection;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    // Material flags.
    // x=unused, y=unused, z=ambient, w=unused
    vec4 materialFlags;
        // Note: per-material data (mapping/specular/triplanar) is stored in the Materials SSBO
        // `materialFlags` kept here for small per-pass overrides (e.g. global normal-mapping toggle in .w)
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x/y/z = unused, w=global shadows enabled (1.0 = on)
    vec4 debugParams; // x=debugMode (0=normal,1=fragment normal,2=normal map (world),3=uv,4=tangent,5=bitangent,6=geometry normal (world),7=albedo,8=normal texture,9=height/bump,10=lighting (N·L,shadow),11=normal from derivatives,12=light vector (rgb),13=N·L grayscale,14=shadow diagnostics)
        // Sky parameters moved to a dedicated Sky UBO (see below)
        vec4 tessParams; // x = tessNearDist, y = tessFarDist, z = tessMinLevel, w = tessMaxLevel
        vec4 passParams;   // x = isShadowPass (1.0 for shadow pass, 0.0 for main pass)
} ubo;

// Packed material data uploaded once to GPU. Matches the CPU-side MaterialGPU (4 vec4s).
// Access this as `materials[texIndex]` from shaders. Uses std430 for tightly-packed vec4 alignment.
struct MaterialGPU {
    vec4 materialFlags;   // .z = ambientFactor
    vec4 mappingParams;   // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    vec4 specularParams;  // x = specularStrength, y = shininess
    vec4 triplanarParams; // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
};

layout(std430, binding = 5) readonly buffer Materials {
    MaterialGPU materials[];
};

// Dedicated UBO for skysphere parameters. Bound separately so sky shaders
// can read a small, focused uniform block instead of the large scene UBO.
layout(binding = 6) uniform SkyUBO {
    vec4 skyHorizon; // rgb = horizon color, a = unused
    vec4 skyZenith;  // rgb = zenith color, a = unused
    vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = unused
    vec4 nightHorizon; // rgb = night horizon color
    vec4 nightZenith;  // rgb = night zenith color
    vec4 nightParams;  // x = night intensity (0..1), y = starIntensity, z/w unused
} sky;