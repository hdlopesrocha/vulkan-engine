#version 450

// Water fragment shader
// Samples scene color with Perlin noise-based refraction, specular lighting, and depth-based effects

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragPosClip;  // clip-space position for scene sampling
layout(location = 4) in vec3 fragDebug;   // debug visual (displacement)

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMask;

// Use the same UBO as main shader
#include "includes/ubo.glsl"

// Water-specific parameters (set 0, binding 7)
layout(set = 0, binding = 7) uniform WaterParamsUBO {
    vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=unused
    vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    vec4 params3;  // x=noiseTimeSpeed, y=waterTime, z=unused, w=unused
    vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    vec4 deepColor; // xyz = deepColor, w = unused
    vec4 waveParams; // x=waveNoiseScale, y=waveNoiseOctaves, z=waveNoisePersistence, w=unused
    vec4 waveParams2; // x=unused, y=unused, z=bumpAmplitude, w=depthFalloff
    vec4 reserved;   // unused
} waterParams;

// Scene color and depth textures for refraction and edge foam (set 2)
layout(set = 2, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 2, binding = 2) uniform sampler2D sceneSkyTex;

// Near/far planes for linearizing depth – read from UBO passParams (z = near, w = far)
// so they always match the glm::perspective call on the CPU side.

// Linearize depth from Vulkan [0,1] depth buffer to eye-space distance.
// With GLM_FORCE_DEPTH_ZERO_TO_ONE the projection maps z_eye to [0,1]:
//   d = f*(z - n) / (z*(f - n))   =>   z = n*f / (f - d*(f - n))
float linearizeDepth(float depth) {
    float nearPlane = ubo.passParams.z;
    float farPlane  = ubo.passParams.w;
    return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
}

#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"

void main() {
    // Get water parameters from UBO
    // Prefer time from water params UBO (reliable for this shader); fallback to global UBO
    float time = waterParams.params3.y;
    if (time == 0.0) time = ubo.passParams.x;
    
    // Water rendering parameters from water UBO
    float refractionStrength = waterParams.params1.x;
    float fresnelPower = waterParams.params1.y;
    float transparency = waterParams.params1.z;
    float waterTint = waterParams.params2.x;
    float noiseScale = waterParams.params2.y;
    int noiseOctaves = int(max(waterParams.params2.z, 1.0));
    float noisePersistence = waterParams.params2.w;
    float noiseTimeSpeed = waterParams.params3.x;


    // Apply noise time speed
    float animTime = time * noiseTimeSpeed;
    
    // Normal used for lighting:
    // Fall back to procedural normal only when tessellation path is inactive/invalid.
    vec3 normal = normalize(fragNormal);

    // Derive the normal from the exact same waterWaveDisplacement() used by the TES
    // so the shading is continuous with the bump surface.
    float eps = 0.5;

    // Same parameters the TES feeds into waterWaveDisplacement
    float bAmp    = waterParams.waveParams2.z;

    // Depth-based wave attenuation (must match the TES depth ramp so
    // the procedural normal is consistent with the displaced geometry).
    float waveDepthTransition = waterParams.shallowColor.w;
    if (waveDepthTransition > 0.0) {
        vec2 earlyScreenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        float earlySceneDepth = texture(sceneDepthTex, earlyScreenUV).r;
        float earlyWaterDepth = gl_FragCoord.z;
        float earlyDepthDiff = max(linearizeDepth(earlySceneDepth) - linearizeDepth(earlyWaterDepth), 0.0);
        bAmp *= smoothstep(0.0, waveDepthTransition, earlyDepthDiff);
    }

    // Build tangent frame from the original mesh normal
    vec3 N  = normal;
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);

    // Finite-difference the displacement along the tangent directions
    float h0 = waterWaveDisplacement(fragPos, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);
    float ht = waterWaveDisplacement(fragPos + eps * T, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);
    float hb = waterWaveDisplacement(fragPos + eps * B, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);

    // Classic bump-map perturbation: N' = N - dh/dT * T - dh/dB * B
    normal = normalize(N - ((ht - h0) / eps) * T - ((hb - h0) / eps) * B);

    
    // Normalize vectors
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPos);
    // Keep the normal facing the visible side to avoid flat/dark lighting from flipped orientation.
    if (dot(normal, viewDir) < 0.0) normal = -normal;
    vec3 lightDir = normalize(-ubo.lightDir.xyz);
    
    // Base screen UV from clip position
    vec2 screenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
    
    // Debug: visual displacement color when debug mode set to 32 ("Water Displacement")
    if (int(ubo.debugParams.x) == 32) {
        // Prefer tessellation-provided debug value when available (fragDebug).
        // But also compute a per-fragment approximation of the bump displacement so the debug
        // mode works even when tessellation is disabled.
        float timeDebug = waterParams.params3.y;
        if (timeDebug == 0.0) timeDebug = ubo.passParams.x;
        float waveScaleDbg = 1.0;  // No longer in passParams (z=nearPlane now)

        float waveNoiseScaleDbg = 1.0 / max(waterParams.waveParams.x, 0.0001);
        int waveNoiseOctavesDbg = int(max(waterParams.waveParams.y, 1.0));
        float waveNoisePersistenceDbg = waterParams.waveParams.z;
        float bumpAmpDbg = waterParams.waveParams2.z;

        float animTimeDbg = timeDebug * waterParams.params3.x;
        float waveDisplacementDbg = waterWaveDisplacement(
            fragPos.xyz,
            animTimeDbg,
            waveNoiseScaleDbg,
            waveNoiseOctavesDbg,
            waveNoisePersistenceDbg,
            bumpAmpDbg,
            waveScaleDbg
        );

        float maxExpected = bumpAmpDbg * waveScaleDbg * 1.5;
        float normDisp = clamp((waveDisplacementDbg / maxExpected) * 0.5 + 0.5, 0.0, 1.0);

        vec3 debugCol = fragDebug;
        // If tessellation wasn't producing a debug value (likely zero), prefer computed color
        if (length(debugCol) < 0.001) debugCol = vec3(normDisp);
        outColor = vec4(debugCol, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    // Debug mode 33: raw scene color — verifies whether the solid pass
    // output actually reaches the water fragment shader.
    if (int(ubo.debugParams.x) == 33) {
        vec2 uv = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        vec3 sc = texture(sceneColorTex, uv).rgb*0.9;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    // Debug mode 34: screen UV — verifies correct clip → UV conversion.
    if (int(ubo.debugParams.x) == 34) {
        vec2 uv = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        outColor = vec4(uv, 0.0, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    // === PERLIN NOISE-BASED REFRACTION ===
    // Generate refraction distortion from shared FBM helper.
    vec2 refractionNoise = waterRefractionNoise(
        fragPos.xyz,
        noiseScale,
        animTime,
        int(noiseOctaves),
        noisePersistence
    );
    
    // Combine noise layers for complex refraction pattern
    vec2 refractionOffset = refractionNoise * refractionStrength;
    
    // Reduce refraction at edges (to avoid sampling outside screen)
    float edgeFade = smoothstep(0.0, 0.1, screenUV.x) * smoothstep(1.0, 0.9, screenUV.x) *
                     smoothstep(0.0, 0.1, screenUV.y) * smoothstep(1.0, 0.9, screenUV.y);
    refractionOffset *= edgeFade;
    
    // Sample scene with refracted UVs
    vec2 refractedUV = screenUV + refractionOffset;
    refractedUV = clamp(refractedUV, 0.001, 0.999);  // Prevent edge artifacts
    
    vec3 sceneColor = texture(sceneColorTex, refractedUV).rgb;
    float sceneDepthRaw = texture(sceneDepthTex, screenUV).r;

    // === DEPTH-BASED EFFECTS ===
    float waterDepthRaw = gl_FragCoord.z;

    // Depth test against solid geometry: in Vulkan depth [0,1], smaller means
    // closer to the camera. So if the solid depth is smaller than water depth,
    // the solid is in front and water must be discarded.
    //
    // Compare in raw depth space (monotonic), with only a tiny epsilon to avoid
    // z-fighting flicker at equal-depth boundaries.
    const float depthEpsilonRaw = 1e-6;
    if (sceneDepthRaw + depthEpsilonRaw < waterDepthRaw) {
        discard;
    }

    float sceneDepthLinear = linearizeDepth(sceneDepthRaw);
    float waterDepthLinear = linearizeDepth(waterDepthRaw);

    float depthDiff = max(sceneDepthLinear - waterDepthLinear, 0.0);
    
    // Depth-based color fade (deeper = more tinted)
    float depthFalloff = waterParams.waveParams2.w;
    if (depthFalloff <= 0.0) depthFalloff = 0.02;
    float depthFade = 1.0 - exp(-depthDiff * depthFalloff);
    
    // === FRESNEL EFFECT ===
    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), fresnelPower);
    fresnel = clamp(fresnel, 0.0, 1.0);
    
    // === SPECULAR LIGHTING (Perlin noise-based) ===
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(normal, halfDir), 0.0);
    
    // Main specular highlight with noise perturbation
    float specNoise = 0.8 + 0.4 * waterFbmNoise(fragPos.xyz, noiseScale, animTime, 1.0,
                                                max(int(noiseOctaves), 1), noisePersistence, vec3(0.0));
    float specular = pow(specAngle, 128.0) * specNoise;
    vec3 specularColor = ubo.lightColor.xyz * specular * 2.0;
    
    // Sun glitter: high-frequency noise-based sparkles
    float glitterNoise = waterFbmNoise(fragPos.xyz, noiseScale * 3.0, animTime, 3.0,
                                       max(int(noiseOctaves) - 2, 1), noisePersistence, vec3(0.0));
    float glitterThreshold = 0.7 + 0.2 * waterFbmNoise(fragPos.xyz, noiseScale * 0.5, animTime, 0.5,
                                                       max(int(noiseOctaves), 1), noisePersistence, vec3(0.0));
    float glitter = smoothstep(glitterThreshold, 1.0, glitterNoise) * pow(specAngle, 32.0);
    specularColor += ubo.lightColor.xyz * glitter * 1.5;
    
    // === REFLECTION ===
    vec3 reflectDir = reflect(-viewDir, normal);
    // Sample sky from equirectangular map using reflection direction
    // Convert 3D direction to equirect UV: x = atan(z,x)/2π+0.5, y = acos(y)/π
    const float PI = 3.14159265358979;
    vec2 skyUV;
    skyUV.x = atan(reflectDir.z, reflectDir.x) / (2.0 * PI) + 0.5;
    skyUV.y = acos(clamp(reflectDir.y, -1.0, 1.0)) / PI;
    vec3 skyColor = texture(sceneSkyTex, skyUV).rgb;
    
    // === WATER COLOR COMPOSITION ===
    // Water tint colors from UBO
    vec3 deepTint = waterParams.deepColor.rgb;
    vec3 shallowTint = waterParams.shallowColor.rgb;
    vec3 waterTintColor = mix(shallowTint, deepTint, depthFade);
    
    // Blend scene color with water tint based on depth.
    // Use a lerp that preserves the scene at shallow depths and transitions
    // to the water tint at greater depths.  The waterTint parameter controls
    // how strongly the water color replaces the scene.
    float tintBlend = clamp(depthFade * waterTint, 0.0, 0.85);
    vec3 refractedColor = mix(sceneColor, waterTintColor, tintBlend);
    
    // Mix refracted color with reflection based on fresnel
    vec3 waterColor = mix(refractedColor, skyColor, fresnel * 0.6);
    
    // Add specular highlights
    waterColor += specularColor;
    
    // === FINAL OUTPUT ===
    float alpha = 1.0;
    outColor = vec4(waterColor, alpha);
    outNormal = vec4(normal, 0.0);
    outMask = vec4(1.0);

   // Debug mode 35: water noise
    if (int(ubo.debugParams.x) == 35) {
        outColor = vec4(refractionNoise, 0.5 + 0.5 * (refractionNoise.x - refractionNoise.y), 1.0);
        return;
    }

    // Debug mode 36: final displaced normal used by shading.
    if (int(ubo.debugParams.x) == 36) {
        vec3 n = normalize(normal);
        outColor = vec4(n * 0.5 + 0.5, 1.0);
        return;
    }

}
