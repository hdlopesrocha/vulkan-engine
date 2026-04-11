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
layout(location = 7) flat in int fragTexIndex;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMask;

// Use the same UBO as main shader
#include "includes/ubo.glsl"
#include "includes/textures.glsl"
#include "includes/shadows.glsl"


// Scene color and depth textures for refraction and edge foam (set 2)
layout(set = 2, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 2, binding = 3) uniform sampler2D waterBackDepthTex;  // back-face depth for volume thickness
layout(set = 2, binding = 4) uniform samplerCube sceneSkyCube;    // solid 360 cubemap (used directly)

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
#include "includes/voronoi.glsl"

void main() {
    // Get water parameters from SSBO indexed by fragment texIndex
    WaterParamsGPU wp = waterParams[fragTexIndex];
    float time = waterRenderUBO.timeParams.x;

    // Water rendering parameters from selected water params
    float refractionStrength = wp.params1.x;
    float fresnelPower = wp.params1.y;
    float transparency = wp.params1.z;
    float waterTint = wp.params2.x;
    float noiseScale = wp.params2.y;
    int noiseOctaves = int(max(wp.params2.z, 1.0));
    float noisePersistence = wp.params2.w;
    float noiseTimeSpeed = wp.params3.x;
    float reflectionStrength = wp.params1.w;
    float specularIntensity = wp.params3.z;
    float specularPowerParam = wp.params3.w;
    float glitterIntensity = wp.deepColor.w;

    // Feature toggles and blur parameters
    bool enableReflection = wp.reserved1.x > 0.5;
    bool enableRefraction = wp.reserved1.y > 0.5;
    bool enableBlur       = wp.reserved1.z > 0.5;
    float blurRadius      = wp.reserved1.w;
    int   blurSamples     = max(int(wp.reserved2.x), 1);
    float volumeBlurRate  = wp.reserved2.y;
    float volumeBumpRate  = wp.reserved2.z;

    // Apply noise time speed
    float animTime = time * noiseTimeSpeed;

    // === WATER VOLUME THICKNESS ===
    // Compute volume thickness from back-face depth (rendered with reversed winding)
    // before the normal computation, so we can modulate bump amplitude.
    vec2 earlyScreenUV0 = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
    float backFaceDepthRaw = texture(waterBackDepthTex, earlyScreenUV0).r;
    float frontFaceLinear  = linearizeDepth(gl_FragCoord.z);
    float backFaceLinear   = linearizeDepth(backFaceDepthRaw);
    float sceneDepthRaw    = texture(sceneDepthTex, earlyScreenUV0).r;
    float sceneDepthEarly  = linearizeDepth(sceneDepthRaw);

    // Clamp to scene depth so thickness doesn't extend beyond solid ground.
    // Use the nearer surface along the view ray, not the farther one.
    float effectiveBack    = min(backFaceLinear, sceneDepthEarly);

    // Reconstruct world-space positions for a true view-ray thickness measurement.
    mat4 invViewProj = inverse(ubo.viewProjection);
    vec4 backFaceWorldH = invViewProj * vec4(earlyScreenUV0 * 2.0 - 1.0, backFaceDepthRaw, 1.0);
    vec3 backFaceWorld = backFaceWorldH.xyz / backFaceWorldH.w;
    vec4 sceneWorldH = invViewProj * vec4(earlyScreenUV0 * 2.0 - 1.0, sceneDepthRaw, 1.0);
    vec3 sceneWorldPos = sceneWorldH.xyz / sceneWorldH.w;

    vec3 worldFrontPos = fragPosWorld;
    vec3 worldRayDir = normalize(worldFrontPos - ubo.viewPos.xyz);
    float backFaceThickness = max(dot(backFaceWorld - worldFrontPos, worldRayDir), 0.0);
    float sceneThickness    = max(dot(sceneWorldPos - worldFrontPos, worldRayDir), 0.0);
    float waterThickness    = min(backFaceThickness, sceneThickness);

    // Depth-based modulation factors (exponential ramp)
        float volumeBlurFactor = (volumeBlurRate > 0.0) ? (1.0 - exp(-waterThickness * volumeBlurRate)) : 1.0;
        float volumeBumpFactor = (volumeBumpRate > 0.0) ? (1.0 - exp(-waterThickness * volumeBumpRate)) : 1.0;
        blurRadius *= volumeBlurFactor;
    
    
    // Common bump parameters.
    float eps = 0.5;
    float bAmp = wp.waveParams.z * volumeBumpFactor;

    // Depth-based wave attenuation (must match the TES depth ramp so
    // the procedural normal is consistent with the displaced geometry).
    float waveDepthTransition = wp.shallowColor.w;
    if (waveDepthTransition > 0.0) {
        vec2 earlyScreenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        float earlySceneDepth = texture(sceneDepthTex, earlyScreenUV).r;
        float earlyWaterDepth = gl_FragCoord.z;
        float earlyDepthDiff = max(linearizeDepth(earlySceneDepth) - linearizeDepth(earlyWaterDepth), 0.0);
        bAmp *= smoothstep(0.0, waveDepthTransition, earlyDepthDiff);
    }

    vec3 N  = normalize(fragNormal);
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);

    vec3 normal;
    if (ubo.passParams.y > 0.5) {
        normal = N;
    } else {
        // When tessellation is inactive, approximate the bumped normal from
        // the procedural displacement on the flat base mesh.
        float h0 = waterWaveDisplacement(fragPos, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);
        float ht = waterWaveDisplacement(fragPos + eps * T, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);
        float hb = waterWaveDisplacement(fragPos + eps * B, animTime, noiseScale, noiseOctaves, noisePersistence, bAmp, 1.0);

        normal = normalize(N - ((ht - h0) / eps) * T - ((hb - h0) / eps) * B);
    }

    
    // Normalize vectors
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    // Keep the normal facing the visible side to avoid flat/dark lighting from flipped orientation.
    if (dot(normal, viewDir) < 0.0) normal = -normal;
    vec3 lightDir = normalize(-ubo.lightDir.xyz);
    
    // Base screen UV from clip position
    vec2 screenUV = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
    
    int dbgMode = int(ubo.debugParams.x + 0.5);

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
    sceneDepthRaw = texture(sceneDepthTex, screenUV).r;

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

    // Depth difference: use the smaller of
    //  - back-face distance to the front water face, and
    //  - solid scene depth distance to the water fragment depth.
    // This prevents the depth-based fades from exceeding the actual
    // available water column determined by either the back-face or
    // occluding solid geometry.
    float backFaceDiff = max(backFaceLinear - frontFaceLinear, 0.0);
    float solidDiff = max(sceneDepthLinear - waterDepthLinear, 0.0);
    float depthDiff = min(backFaceDiff, solidDiff);
    
    // Depth-based color fade (deeper = more tinted)
    float depthFalloff = wp.waveParams.w;
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

    vec3 skyColor = texture(sceneSkyCube, reflectDir).rgb;

    // Uniform reflection toggle: when set, apply reflectionStrength uniformly
    // instead of modulating by Fresnel. This flag is stored in reserved2.w
    // (see WaterParamsGPU.reserved2.w).
    bool uniformReflection = wp.reserved2.w > 0.5;


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
    vec3 deepTint = wp.deepColor.rgb;
    vec3 shallowTint = wp.shallowColor.rgb;


    // Caustic parameters
    vec3 causticColor = wp.causticColor.rgb;
    float causticScale = wp.causticParams.x;
    float causticIntensity = wp.causticParams.y;
    float causticPower = wp.causticParams.z;
    int causticType = int(round(clamp(wp.causticExtraParams.z, 0.0, 1.0)));
    float causticVelocity = wp.causticExtraParams.w;
    float causticAnimTime = animTime * causticVelocity;

    // Compute a volume-based factor from measured water thickness.
    // Use causticParams.w (depth-scale) as a per-layer reference distance; if unset,
    // fall back to a small stable value so division is safe.
    float tintDepthScale = max(wp.causticParams.w, 0.0001);
    float volumeFactor = clamp(waterThickness / tintDepthScale, 0.0, 1.0);

    // Water tint color transitions from shallow → deep depending on volume.
    vec3 waterTintColor = mix(shallowTint, deepTint, volumeFactor);

    // Attenuate tint effect when the measured thickness is very small (near zero).
    // This reduces color influence when the front and back faces are nearly coincident.
    float thicknessAttenuation = smoothstep(0.0, max(0.005, tintDepthScale * 0.25), waterThickness);

    // Blend scene color with water tint based on both the depth-based fade (depthFade)
    // and the measured volume. The `waterTint` parameter scales overall tint strength.
    float tintBlend = clamp(depthFade * waterTint * volumeFactor * thicknessAttenuation, 0.0, 0.85);
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
    float causticDepthScale = wp.causticParams.w; // w = depth-scale (world units)
    float depthInfluence = (causticDepthScale > 0.0) ? clamp(waterThickness / causticDepthScale, 0.0, 1.0) : 1.0;

    // Back-face (bottom) sampling: march along the view ray from the front
    // position by the measured water thickness to approximate the bottom
    // world position — used by both caustic modes.
    vec3 backPos = fragPosWorld + worldRayDir * waterThickness;

    // Line-shaped measure parameters
    float lineScale = wp.causticExtraParams.x;
    float lineMix = clamp(wp.causticExtraParams.y, 0.0, 1.0);

    // Prepare outputs that debug and later code expect
    float caustFront = 0.0;
    float caustBack = 0.0;
    float lineFrontRaw = 0.0;
    float lineBackRaw = 0.0;
    float cloudFinal = 0.0;
    float lineFinal = 0.0;
    float lineCombined = 0.0;

    // Compute only the selected caustic noise per-fragment
    if (causticType == 1) {
        // VORONOI-based measures (Worley noise) — jitter feature points using FBM
        vec2 vorFront = voronoi3d(fragPos * causticScale, causticAnimTime, noiseScale, 0.5, noiseOctaves, noisePersistence);
        vec2 vorBack  = voronoi3d(backPos * causticScale, causticAnimTime, noiseScale, 0.5, noiseOctaves, noisePersistence);
        float f1f = vorFront.x;
        float f2f = vorFront.y;
        float f1b = vorBack.x;
        float f2b = vorBack.y;

        caustFront = max(1.0 - f1f, 0.0);
        caustBack  = max(1.0 - f1b, 0.0);
        lineFrontRaw = max(1.0 - (f2f - f1f) * lineScale, 0.0);
        lineBackRaw  = max(1.0 - (f2b - f1b) * lineScale, 0.0);

        // Compose final cloud/line measures and apply power/intensity
        float cloudCombined = mix(caustFront, caustBack, depthInfluence);
        cloudFinal = pow(max(cloudCombined, 1e-6), causticPower);
        lineCombined = mix(lineFrontRaw, lineBackRaw, depthInfluence);
        lineFinal = pow(max(lineCombined, 1e-6), causticPower);
    } else {
        // PERLIN-based measures (existing Jacobian method)
        vec2 ref0 = waterRefractionNoise(fragPos.xyz, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 refT = waterRefractionNoise(fragPos + eps * T, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 refB = waterRefractionNoise(fragPos + eps * B, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 ddT = (refT - ref0) / eps;
        vec2 ddB = (refB - ref0) / eps;
        float detJFront = ddT.x * ddB.y - ddT.y * ddB.x;
        float trFront = ddT.x + ddB.y;
        float anisFront = sqrt(max(trFront * trFront - 4.0 * detJFront, 0.0));

        vec2 ref0b = waterRefractionNoise(backPos.xyz, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 refTb = waterRefractionNoise(backPos + eps * T, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 refBb = waterRefractionNoise(backPos + eps * B, noiseScale, causticAnimTime, int(noiseOctaves), noisePersistence) * refractionStrength;
        vec2 ddTb = (refTb - ref0b) / eps;
        vec2 ddBb = (refBb - ref0b) / eps;
        float detJBack = ddTb.x * ddBb.y - ddTb.y * ddBb.x;
        float trBack = ddTb.x + ddBb.y;
        float anisBack = sqrt(max(trBack * trBack - 4.0 * detJBack, 0.0));

        caustFront = max(-detJFront * causticScale, 0.0);
        caustBack  = max(-detJBack  * causticScale, 0.0);
        lineFrontRaw  = max(anisFront * causticScale * lineScale, 0.0);
        lineBackRaw   = max(anisBack  * causticScale * lineScale, 0.0);

        // Compose final cloud/line measures and apply power/intensity
        float cloudCombined = mix(caustFront, caustBack, depthInfluence);
        cloudFinal = pow(max(cloudCombined, 1e-6), causticPower);
        lineCombined = mix(lineFrontRaw, lineBackRaw, depthInfluence);
        lineFinal = pow(max(lineCombined, 1e-6), causticPower);
    }

    // Blend cloud vs line patterns, then apply intensity and modulations
    float caustRaw = mix(cloudFinal, lineFinal, lineMix);
    float caustic = caustRaw * causticIntensity * depthRampCaust * angularCaust * edgeFade * (1.0 - shadow);

    waterColor += causticColor * caustic;

    // === FINAL OUTPUT ===
    float alpha = 1.0;
    outColor = vec4(waterColor, alpha);

    // Debug: visual displacement color when debug mode set to 32 ("Water Displacement")
    if (dbgMode == 32) {
        // Prefer tessellation-provided debug value when available (fragDebug).
        // But also compute a per-fragment approximation of the bump displacement so the debug
        // mode works even when tessellation is disabled.
        float timeDebug = waterRenderUBO.timeParams.x;
        float waveScaleDbg = 1.0;  // No longer in passParams (z=nearPlane now)

        float bumpAmpDbg = wp.waveParams.z;

        float animTimeDbg = timeDebug * wp.params3.x;
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
    }

    // Debug mode 33: raw scene color — verifies whether the solid pass
    // output actually reaches the water fragment shader.
    if (dbgMode == 33) {
        vec2 uv = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        vec3 sc = texture(sceneColorTex, uv).rgb*0.9;
        outColor = vec4(sc, 1.0);
    }

    // Debug mode 34: screen UV — verifies correct clip → UV conversion.
    if (dbgMode == 34) {
        vec2 uv = (fragPosClip.xy / fragPosClip.w) * 0.5 + 0.5;
        outColor = vec4(uv, 0.0, 1.0);
    }


   // Debug mode 35: water noise
    if (int(ubo.debugParams.x) == 35) {
        outColor = vec4(refractionNoise, 0.5 + 0.5 * (refractionNoise.x - refractionNoise.y), 1.0);
    }

    // Debug mode 36: final displaced normal used by shading.
    if (int(ubo.debugParams.x) == 36) {
        vec3 n = normalize(normal);
        outColor = vec4(n * 0.5 + 0.5, 1.0);
    }

    // --- Reflection sampling debug helpers ---
    // Use the global debug mode (ubo.debugParams.x) to visualize reflection
    // computation steps and cubemap sampling. Helpful to diagnose orientation.
    if (dbgMode == 37) {
        // Visualize reflection vector (packed to [0,1])
        vec3 vis = reflectDir * 0.5 + 0.5;
        outColor = vec4(vis, 1.0);
    }
    if (dbgMode == 38) {
        // Show direct cubemap sample
        outColor = vec4(skyColor, 1.0);
    }

    if (dbgMode == 39) {
        vec3 maps = vec3(clamp(caustFront, 0.0, 1.0), clamp(caustBack, 0.0, 1.0), clamp(mix(caustFront, caustBack, depthInfluence), 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 40) {
        vec3 maps = vec3(clamp(lineFrontRaw, 0.0, 1.0), clamp(lineBackRaw, 0.0, 1.0), clamp(lineCombined, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 41) {
        vec3 maps = vec3(clamp(cloudFinal, 0.0, 1.0), clamp(lineFinal, 0.0, 1.0), clamp(caustRaw, 0.0, 1.0));
        outColor = vec4(maps, 1.0);
    }
    if (dbgMode == 42) {
        outColor = vec4(vec3(clamp(caustic, 0.0, 1.0)), 1.0);
    }

    // --- Water thickness / depth debug modes (43..48) ---
    // 43: Back-face raw depth (texture sample)
    if (dbgMode == 43) {
        outColor = vec4(vec3(backFaceDepthRaw), 1.0);
    }
    // 44: Front-face linear depth (normalized to [0,1])
    if (dbgMode == 44) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((frontFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 45: Back-face linear depth (normalized to [0,1])
    if (dbgMode == 45) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((backFaceLinear - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 46: Scene depth at early-screen UV (linearized, normalized)
    if (dbgMode == 46) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((sceneDepthEarly - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 47: Effective back depth (min(backFaceLinear, sceneDepthEarly))
    if (dbgMode == 47) {
        float nearP = ubo.passParams.z;
        float farP = ubo.passParams.w;
        float v = clamp((effectiveBack - nearP) / max(farP - nearP, 1e-6), 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }
    // 48: Water thickness (normalized by per-layer caustic depth scale or 1.0)
    if (dbgMode == 48) {
        float denom = max(wp.causticParams.w, 1.0);
        float v = clamp(waterThickness / denom, 0.0, 1.0);
        outColor = vec4(vec3(v), 1.0);
    }


    outNormal = vec4(normal, 0.0);
    outMask = vec4(1.0);


}
