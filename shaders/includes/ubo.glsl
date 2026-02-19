// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 viewProjection; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(set = 0, binding = 0) uniform UBO {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 materialFlags;
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x/y/z = unused, w=global shadows enabled (1.0 = on)
    vec4 debugParams; // x=debugMode (0=normal,1=fragment normal,2=normal map (world),3=uv,4=tangent,5=bitangent,6=geometry normal (world),7=albedo,8=normal texture,9=height/bump,10=lighting (N·L,shadow),11=normal from derivatives,12=light vector (rgb),13=N·L grayscale,14=shadow diagnostics,15=triplanar weights,16=tex indices (RGB),17=barycentric weights,18=albedo samples (R/G/B),19=triplanar albedo,20=per-projection triplanar heights (RGB),21=UV vs triplanar height diff,22=triplanar normal,23=per-projection triplanar normals (RGB),24=UV vs triplanar normal diff,25=triplanar bump (height),26=per-projection triplanar bump (RGB),27=UV vs triplanar bump diff)
    vec4 triplanarSettings;
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
    vec4 normalParams;   // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z/w = reserved
};

layout(std430, set = 0, binding = 5) readonly buffer Materials {
    MaterialGPU materials[];
};

// Per-draw model matrices for indirect rendering
// Models SSBO removed — shaders use identity models

// Dedicated UBO for skysphere parameters. Bound separately so sky shaders
// can read a small, focused uniform block instead of the large scene UBO.
layout(set = 0, binding = 6) uniform SkyUBO {
    vec4 skyHorizon; // rgb = horizon color, a = unused
    vec4 skyZenith;  // rgb = zenith color, a = unused
    vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = skyMode (0=gradient, 1=grid)
    vec4 nightHorizon; // rgb = night horizon color
    vec4 nightZenith;  // rgb = night zenith color
    vec4 nightParams;  // x = night intensity (0..1), y = starIntensity, z/w unused
} sky;