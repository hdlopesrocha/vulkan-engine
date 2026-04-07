#version 450

// Water fragment shader
// Samples scene color with Perlin noise-based refraction, specular lighting, and depth-based effects

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragPosClip;  // clip-space position for scene sampling
layout(location = 4) in vec3 fragDebug;   // debug visual (displacement)
layout(location = 5) in vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = 6) in vec4 fragPosLightSpace; // light-space pos (cascade 0)

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMask;

// Use the same UBO as main shader
#include "includes/ubo.glsl"
#include "includes/textures.glsl"
#include "includes/shadows.glsl"

// Water-specific parameters (set 0, binding 7)
layout(set = 0, binding = 7) uniform WaterParamsUBO {
    vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    vec4 params3;  // x=noiseTimeSpeed, y=waterTime, z=specularIntensity, w=specularPower
    vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    vec4 deepColor; // xyz = deepColor, w = glitterIntensity
    vec4 waveParams; // x=unused, y=unused, z=bumpAmplitude, w=depthFalloff
    vec4 reserved1;  // unused
    vec4 reserved2;  // unused
    vec4 reserved3;  // unused (x = cube360Available)
    vec4 causticColor; // rgb = caustic tint, w = unused
    vec4 causticParams; // x = scale, y = intensity, z = power, w = depthScale
    vec4 causticExtraParams; // x = lineScale, y = lineMix, z/w = unused
} waterParams;

// Scene color and depth textures for refraction and edge foam (set 2)
layout(set = 2, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 2, binding = 2) uniform sampler2D sceneSkyTex;
layout(set = 2, binding = 3) uniform sampler2D waterBackDepthTex;  // back-face depth for volume thickness
layout(set = 2, binding = 4) uniform samplerCube sceneSkyCube;    // optional solid 360 cubemap

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
    float reflectionStrength = waterParams.params1.w;
    float specularIntensity = waterParams.params3.z;
    float specularPowerParam = waterParams.params3.w;
    float glitterIntensity = waterParams.deepColor.w;

    // Feature toggles and blur parameters
    bool enableReflection = waterParams.reserved1.x > 0.5;
    bool enableRefraction = waterParams.reserved1.y > 0.5;
    bool enableBlur       = waterParams.reserved1.z > 0.5;
    float blurRadius      = waterParams.reserved1.w;
    int   blurSamples     = max(int(waterParams.reserved2.x), 1);
    float volumeBlurRate  = waterParams.reserved2.y;
    float volumeBumpRate  = waterParams.reserved2.z;

    // Apply noise time speed
    float animTime = time * noiseTimeSpeed;

    // === WATER VOLUME THICKNESS ===
    // Compute volume thickness from back-face depth (rendered with reversed winding)
    // before the normal computation, so we can modulate bump amplitude.
    vec2 earlyScreenUV0 = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
    float backFaceDepthRaw = texture(waterBackDepthTex, earlyScreenUV0).r;
    float frontFaceLinear  = linearizeDepth(gl_FragCoord.z);
    float backFaceLinear   = linearizeDepth(backFaceDepthRaw);
    // Clamp to scene depth so thickness doesn't extend beyond solid ground
    float sceneDepthEarly  = linearizeDepth(texture(sceneDepthTex, earlyScreenUV0).r);
    float effectiveBack    = min(backFaceLinear, sceneDepthEarly);
    float waterThickness   = max(effectiveBack - frontFaceLinear, 0.0);

    // Depth-based modulation factors (exponential ramp)
    float volumeBlurFactor = (volumeBlurRate > 0.0) ? (1.0 - exp(-waterThickness * volumeBlurRate)) : 1.0;
    float volumeBumpFactor = (volumeBumpRate > 0.0) ? (1.0 - exp(-waterThickness * volumeBumpRate)) : 1.0;

    // Modulate blur radius by volume thickness
    blurRadius *= volumeBlurFactor;
    
    // Caustic parameters
    vec3 causticColor = waterParams.causticColor.rgb;
    float causticScale = waterParams.causticParams.x;
    float causticIntensity = waterParams.causticParams.y;
    float causticPower = waterParams.causticParams.z;
    
    // Normal used for lighting:
    // Fall back to procedural normal only when tessellation path is inactive/invalid.
    vec3 normal = normalize(fragNormal);

    // Derive the normal from the exact same waterWaveDisplacement() used by the TES
    // so the shading is continuous with the bump surface.
    float eps = 0.5;

    // Same parameters the TES feeds into waterWaveDisplacement
    // Modulate bump amplitude by volume thickness
    float bAmp    = waterParams.waveParams.z * volumeBumpFactor;

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

        float bumpAmpDbg = waterParams.waveParams.z;

        float animTimeDbg = timeDebug * waterParams.params3.x;
        float waveDisplacementDbg = waterWaveDisplacement(
            fragPos.xyz,
            animTimeDbg,
            noiseScale,
            noiseOctaves,
            noisePersistence,
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
    vec2 refractionOffset = enableRefraction
        ? refractionNoise * refractionStrength
        : vec2(0.0);
    
    // Reduce refraction at edges (to avoid sampling outside screen)
    float edgeFade = smoothstep(0.0, 0.1, screenUV.x) * smoothstep(1.0, 0.9, screenUV.x) *
                     smoothstep(0.0, 0.1, screenUV.y) * smoothstep(1.0, 0.9, screenUV.y);
    refractionOffset *= edgeFade;
    
    // Sample scene with refracted UVs
    vec2 refractedUV = screenUV + refractionOffset;
    refractedUV = clamp(refractedUV, 0.001, 0.999);  // Prevent edge artifacts
    
    // === SCENE COLOR SAMPLING (optional PCF blur) ===
    vec3 sceneColor;
    if (enableBlur && blurSamples > 1) {
        // PCF-style NxN box blur on the refracted scene color
        vec2 texelSize = 1.0 / textureSize(sceneColorTex, 0);
        vec3 colorAccum = vec3(0.0);
        int halfK = blurSamples / 2;
        float totalWeight = 0.0;
        for (int bx = -halfK; bx <= halfK; ++bx) {
            for (int by = -halfK; by <= halfK; ++by) {
                vec2 offset = vec2(float(bx), float(by)) * texelSize * blurRadius;
                vec2 sampleUV = clamp(refractedUV + offset, 0.001, 0.999);
                colorAccum += texture(sceneColorTex, sampleUV).rgb;
                totalWeight += 1.0;
            }
        }
        sceneColor = colorAccum / totalWeight;
    } else {
        sceneColor = texture(sceneColorTex, refractedUV).rgb;
    }
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
    float depthFalloff = waterParams.waveParams.w;
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
    float specular = pow(specAngle, specularPowerParam) * specNoise;
    vec3 specularColor = ubo.lightColor.xyz * specular * specularIntensity;
    
    // Sun glitter: high-frequency noise-based sparkles
    float glitterNoise = waterFbmNoise(fragPos.xyz, noiseScale * 3.0, animTime, 3.0,
                                       max(int(noiseOctaves) - 2, 1), noisePersistence, vec3(0.0));
    float glitterThreshold = 0.7 + 0.2 * waterFbmNoise(fragPos.xyz, noiseScale * 0.5, animTime, 0.5,
                                                       max(int(noiseOctaves), 1), noisePersistence, vec3(0.0));
    float glitter = smoothstep(glitterThreshold, 1.0, glitterNoise) * pow(specAngle, 32.0);
    specularColor += ubo.lightColor.xyz * glitter * glitterIntensity;
    
    // === REFLECTION ===
    vec3 reflectDir = reflect(-viewDir, normal);
    // Sample sky from equirectangular map using reflection direction
    // Convert 3D direction to equirect UV: x = atan(z,x)/2π+0.5, y = acos(y)/π
    const float PI = 3.14159265358979;
    vec2 skyUV;
    skyUV.x = atan(reflectDir.z, reflectDir.x) / (2.0 * PI) + 0.5;
    skyUV.y = acos(clamp(reflectDir.y, -1.0, 1.0)) / PI;
    // Prefer cubemap sampling when available (set via WaterParamsGPU.reserved3.x)
    bool cubeAvailable = waterParams.reserved3.x > 0.5;
    vec3 skyColor;
        if (cubeAvailable) { 
        // Mirror the -X cubemap region horizontally to match cube face orientation
        vec3 rd = reflectDir;
        float axc = abs(rd.x);
        float ayc = abs(rd.y);
        float azc = abs(rd.z);
        // +Y face: mirror horizontally by flipping X
        if (ayc >= axc && ayc >= azc && rd.y > 0.0) {
            rd.x = -rd.x;
        }
        // +X face: mirror horizontally by flipping Z
        else if (axc >= ayc && axc >= azc && rd.x > 0.0) {
            rd.z = -rd.z;
        }
        // -X face: mirror horizontally by flipping Z
        else if (axc >= ayc && axc >= azc && rd.x < 0.0) {
            rd.z = -rd.z;
        }
        // +Z face: mirror horizontally by flipping X
        else if (azc >= axc && azc >= ayc && rd.z > 0.0) {
            rd.x = -rd.x;
        }
        // -Z face: mirror horizontally by flipping X
        else if (azc >= axc && azc >= ayc && rd.z < 0.0) {
            rd.x = -rd.x;
        }
        skyColor = texture(sceneSkyCube, rd).rgb;
    } else {
        skyColor = texture(sceneSkyTex, skyUV).rgb;
    }

    // Uniform reflection toggle: when set, apply reflectionStrength uniformly
    // instead of modulating by Fresnel. This flag is stored in reserved2.w
    // (see WaterParamsGPU.reserved2.w).
    bool uniformReflection = waterParams.reserved2.w > 0.5;

    // --- Reflection sampling debug helpers ---
    // Use the global debug mode (ubo.debugParams.x) to visualize reflection
    // computation steps and alternate sampling conventions without recompiling
    // shaders. Helpful to diagnose equirect/cubemap orientation issues.
    int dbgMode = int(ubo.debugParams.x + 0.5);
    if (dbgMode == 37) {
        // Visualize reflection vector (packed to [0,1])
        vec3 vis = reflectDir * 0.5 + 0.5;
        outColor = vec4(vis, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 38) {
        // Visualize computed equirect UV
        outColor = vec4(skyUV, 0.0, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 39) {
        // Show direct sample from the equirect texture (current behavior)
        vec3 sc = texture(sceneSkyTex, skyUV).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 40) {
        // Sample with flipped X
        vec2 uvx = vec2(1.0 - skyUV.x, skyUV.y);
        vec3 sc = texture(sceneSkyTex, uvx).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 41) {
        // Sample with flipped Y
        vec2 uvy = vec2(skyUV.x, 1.0 - skyUV.y);
        vec3 sc = texture(sceneSkyTex, uvy).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 42) {
        // Alternate mapping: swap X/Z in atan() (useful for diagnosing axis swaps)
        vec2 altUV;
        altUV.x = atan(reflectDir.x, reflectDir.z) / (2.0 * PI) + 0.5;
        altUV.y = acos(clamp(reflectDir.y, -1.0, 1.0)) / PI;
        vec3 sc = texture(sceneSkyTex, altUV).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    // Additional alternate mappings for further diagnosis (43..48)
    if (dbgMode == 43) {
        // Swap X/Z (dir = [z, y, x])
        vec3 d = vec3(reflectDir.z, reflectDir.y, reflectDir.x);
        vec2 uv = vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
        vec3 sc = texture(sceneSkyTex, uv).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 44) {
        // Swap X/Z and negate Z (dir = [-z, y, x])
        vec3 d = vec3(-reflectDir.z, reflectDir.y, reflectDir.x);
        vec2 uv = vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
        vec3 sc = texture(sceneSkyTex, uv).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 45) {
        // Negate Z component
        vec3 d = vec3(reflectDir.x, reflectDir.y, -reflectDir.z);
        vec2 uv = vec2(atan(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
        vec3 sc = texture(sceneSkyTex, uv).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 46) {
        // Rotate UV (clockwise 90 deg)
        vec2 uvrot = vec2(1.0 - skyUV.y, skyUV.x);
        vec3 sc = texture(sceneSkyTex, uvrot).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 47) {
        // Rotate UV (counter-clockwise 90 deg)
        vec2 uvrot2 = vec2(skyUV.y, 1.0 - skyUV.x);
        vec3 sc = texture(sceneSkyTex, uvrot2).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgMode == 48) {
        // Invert both UV axes
        vec2 uvinv = vec2(1.0 - skyUV.x, 1.0 - skyUV.y);
        vec3 sc = texture(sceneSkyTex, uvinv).rgb;
        outColor = vec4(sc, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    // === SHADOW ON WATER ===
    float shadow = 0.0;
    if (ubo.shadowEffects.w > 0.5) {
        float NdotL = max(dot(normal, lightDir), 0.0);
        if (NdotL > 0.01) {
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(fragPosLightSpace, bias);
        } else {
            shadow = 1.0;
        }
    }
    
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
    
    // Mix refracted color with reflection. By default, use Fresnel weighting
    // to increase reflection at grazing angles. If `uniformReflection` is
    // enabled, use `reflectionStrength` directly so reflection appears
    // across all pixels uniformly (useful for debugging/stylized look).
    vec3 waterColor;
    if (enableReflection) {
        float reflMix = uniformReflection ? reflectionStrength : (fresnel * reflectionStrength);
        waterColor = mix(refractedColor, skyColor, reflMix);
    } else {
        waterColor = refractedColor;
    }
    
    // Add specular highlights (suppressed in shadow)
    waterColor += specularColor * (1.0 - shadow);

    // Darken diffuse water color in shadow
    waterColor *= mix(1.0, 0.55, shadow);

    // (Volume light accumulation removed — caustics only)

    // === CAUSTICS / LIGHT FOCUSING ===
    // Estimate local Jacobian of the refraction offset field by finite-difference
    // along the surface tangent frame (T,B). Negative determinant indicates
    // local focusing (area contraction) which produces brighter caustics.
    // Compute incidence/angle and a simple depth-based ramp for caustic strength
    // Incidence term for caustic modulation
    float lightIncidenceCaust = max(dot(normal, lightDir), 0.0);
    float angularCaust = (causticPower > 0.0) ? pow(lightIncidenceCaust, causticPower) : 1.0;

    // Depth-based ramps: keep a small exponential ramp as an additional softening
    float depthRampCaust = 1.0 - exp(-waterThickness * 0.02);

    // Volume-aware caustics: evaluate the refraction noise Jacobian at both
    // the front surface and at the back-face (bottom) and blend according
    // to water thickness. This approximates how focusing changes through the
    // water column and lets caustics appear where the volume causes stronger
    // focusing on the bottom.
    float causticDepthScale = waterParams.causticParams.w; // w = depth-scale (world units)
    float depthInfluence = (causticDepthScale > 0.0) ? clamp(waterThickness / causticDepthScale, 0.0, 1.0) : 1.0;

    // Front-surface Jacobian (same as before)
    vec2 ref0 = waterRefractionNoise(fragPos.xyz, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 refT = waterRefractionNoise(fragPos + eps * T, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 refB = waterRefractionNoise(fragPos + eps * B, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 ddT = (refT - ref0) / eps;
    vec2 ddB = (refB - ref0) / eps;
    float detJFront = ddT.x * ddB.y - ddT.y * ddB.x;
    float trFront = ddT.x + ddB.y;
    float anisFront = sqrt(max(trFront * trFront - 4.0 * detJFront, 0.0));

    // Back-face (bottom) sampling: march along the view ray from the front
    // position by the measured water thickness to approximate the bottom
    // world position, then compute the Jacobian there.
    vec3 rayDirWorld = normalize(fragPos - ubo.viewPos.xyz); // camera->fragment ray
    vec3 backPos = fragPos + rayDirWorld * waterThickness;

    vec2 ref0b = waterRefractionNoise(backPos.xyz, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 refTb = waterRefractionNoise(backPos + eps * T, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 refBb = waterRefractionNoise(backPos + eps * B, noiseScale, animTime, int(noiseOctaves), noisePersistence) * refractionStrength;
    vec2 ddTb = (refTb - ref0b) / eps;
    vec2 ddBb = (refBb - ref0b) / eps;
    float detJBack = ddTb.x * ddBb.y - ddTb.y * ddBb.x;
    float trBack = ddTb.x + ddBb.y;
    float anisBack = sqrt(max(trBack * trBack - 4.0 * detJBack, 0.0));

    // Cloudy caustic measure (area contraction)
    float caustFront = max(-detJFront * causticScale, 0.0);
    float caustBack  = max(-detJBack  * causticScale, 0.0);
    float cloudCombined = mix(caustFront, caustBack, depthInfluence);
    float cloudFinal = pow(max(cloudCombined, 1e-6), causticPower);

    // Line-shaped measure from anisotropy (difference of eigenvalues)
    float lineScale = waterParams.causticExtraParams.x;
    float lineMix = clamp(waterParams.causticExtraParams.y, 0.0, 1.0);
    float lineFrontRaw = max(anisFront * causticScale * lineScale, 0.0);
    float lineBackRaw  = max(anisBack  * causticScale * lineScale, 0.0);
    float lineCombined = mix(lineFrontRaw, lineBackRaw, depthInfluence);
    float lineFinal = pow(max(lineCombined, 1e-6), causticPower);

    // Blend cloud vs line patterns, then apply intensity and modulations
    float caustRaw = mix(cloudFinal, lineFinal, lineMix);
    float caustic = caustRaw * causticIntensity * depthRampCaust * angularCaust * edgeFade * (1.0 - shadow);

    // --- Debug visualizations for caustics ---
    // 49: Area-contraction maps (R=front, G=back, B=blended)
    // 50: Anisotropy maps (R=front, G=back, B=blended)
    // 51: Cloud/Line components (R=cloud, G=line, B=pre-mix caustRaw)
    // 52: Final caustic mask (post-modulation, clamped)
    int dbgModeCaust = int(ubo.debugParams.x + 0.5);
    if (dbgModeCaust == 49) {
        vec3 maps = vec3(clamp(caustFront, 0.0, 1.0), clamp(caustBack, 0.0, 1.0), clamp(mix(caustFront, caustBack, depthInfluence), 0.0, 1.0));
        outColor = vec4(maps, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgModeCaust == 50) {
        vec3 maps = vec3(clamp(lineFrontRaw, 0.0, 1.0), clamp(lineBackRaw, 0.0, 1.0), clamp(lineCombined, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgModeCaust == 51) {
        vec3 maps = vec3(clamp(cloudFinal, 0.0, 1.0), clamp(lineFinal, 0.0, 1.0), clamp(caustRaw, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }
    if (dbgModeCaust == 52) {
        outColor = vec4(vec3(clamp(caustic, 0.0, 1.0)), 1.0);
        outNormal = vec4(normal, 0.0);
        outMask = vec4(1.0);
        return;
    }

    waterColor += causticColor * caustic;

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
