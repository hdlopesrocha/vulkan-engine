
#version 450
#include "includes/locations.glsl"

layout(location = VARY_COLOR) in vec3 fragColor;
layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) in vec3 fragTexWeights;
layout(location = VARY_SHARPNORMAL) in vec3 fragSharpNormal; // face normal computed in TES (sharp)

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

#ifndef BRUSH_PASS
layout(set = 1, binding = 0) uniform sampler2D brushDepthTex;
layout(set = 1, binding = 1) uniform sampler2D brushBackFaceDepthTex;
#endif

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"

#include "includes/shadows.glsl"

// Global toggles
bool roughnessEnabled = ubo.debugParams.y > 0.5;
bool aoEnabled = ubo.debugParams.z > 0.5;

void main() {
    bool isShadowPass = ubo.passParams.x > 0.5;
    // Provide a default color so early debug/special-case returns still
    // produce a valid output for downstream passes.
    outColor = vec4(0.0);

    // Fast-path for shadow pass: skip expensive lighting/texture work.
    if (isShadowPass) {
        outColor = vec4(0.0);
        return;
    }

    // Texture indices, optionally overridden by PAINT/REMOVE mode
    ivec3 texIndices = fragTexIndices;
    float brushRedFade = 0.0;
#ifndef BRUSH_PASS
    bool isPaintMode = ubo.brushParams.y > 1.5;
    bool isRemoveMode = ubo.brushParams.y > 0.5 && ubo.brushParams.y < 1.5;
    if (isPaintMode || isRemoveMode) {
        vec2 brushUV = gl_FragCoord.xy / vec2(textureSize(brushDepthTex, 0));
        float brushFront = texture(brushDepthTex, brushUV).r;
        float brushBack = texture(brushBackFaceDepthTex, brushUV).r;
        float fragDepth = gl_FragCoord.z;
        if (fragDepth >= brushFront && fragDepth <= brushBack) {
            int brushTexIndex = int(ubo.brushParams.x + 0.5);
            texIndices = ivec3(brushTexIndex);
            if (isRemoveMode) {
                brushRedFade = (sin(ubo.brushParams.w * 6.28318) + 1.0) * 0.5;
            }
        }
    }
#endif

    // Use the three tex indices and barycentric weights provided by the TES for blending
    vec3 w = fragTexWeights;
    vec2 uv = fragUV;
    bool usedTriplanar = false;

    // Geometry normal (world-space) and geometric face normal from derivatives
    vec3 N = normalize(fragNormal);
    vec3 geomN = normalize(cross(dFdx(fragPosWorld), dFdy(fragPosWorld)));
    if (length(geomN) < 1e-5) geomN = N;
    vec3 worldNormal = N;

    // Precompute triplanar blend weights from geometric normal (abs^2 normalized)
    // Compute triplanar weights with a configurable dead-zone threshold and adjustable steepness
    vec3 triW = abs(geomN);

    // Subtract threshold and clamp so small components remain zero until threshold is exceeded
    float t = ubo.triplanarSettings.x; // threshold (0..1)
    vec3 wt = max(vec3(0.0), triW - vec3(t));

    // Apply exponent to make transitions steeper
    float e = max(1.0, ubo.triplanarSettings.y);
    wt = pow(wt, vec3(e));
    float triWSum = wt.x + wt.y + wt.z + 1e-6;
    triW = wt / triWSum;
    
    // Sample albedo texture (triplanar when enabled)
    vec3 albedoColor;
    vec3 tripNormal0;
    vec3 tripNormal1;
    vec3 tripNormal2;

    // Mix triplanar flag across the three materials
    float triFlag = dot(vec3(materials[texIndices.x].triplanarParams.z, materials[texIndices.y].triplanarParams.z, materials[texIndices.z].triplanarParams.z), w);

    // Compute triplanar UVs once per material and reuse across all map fetches below.
    vec2 uv0X, uv0Y, uv0Z;
    vec2 uv1X, uv1Y, uv1Z;
    vec2 uv2X, uv2Y, uv2Z;
    if (w.x > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.x, geomN, uv0X, uv0Y, uv0Z);
    if (w.y > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.y, geomN, uv1X, uv1Y, uv1Z);
    if (w.z > 0.0) computeTriplanarUVs(fragPosWorldNotDisplaced, texIndices.z, geomN, uv2X, uv2Y, uv2Z);

    if (triFlag > 0.5) {
        usedTriplanar = true;
        // compute triplanar albedo per-layer then blend
        vec3 a0 = w.x > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : vec3(0.0);
        vec3 a1 = w.y > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : vec3(0.0);
        vec3 a2 = w.z > 0.0 ? computeTriplanarAlbedoUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : vec3(0.0);
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
        // If normal mapping/triplanar normal enabled per-material or global, compute blended triplanar normal
        float mapFlag0 = materials[texIndices.x].mappingParams.x;
        float mapFlag1 = materials[texIndices.y].mappingParams.x;
        float mapFlag2 = materials[texIndices.z].mappingParams.x;
        if ((mapFlag0 * w.x + mapFlag1 * w.y + mapFlag2 * w.z) > 0.5 || ubo.materialFlags.w > 0.5) {
            tripNormal0 = w.x > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.x, N, uv0X, uv0Y, uv0Z) : vec3(0.0);
            tripNormal1 = w.y > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.y, N, uv1X, uv1Y, uv1Z) : vec3(0.0);
            tripNormal2 = w.z > 0.0 ? computeTriplanarNormalUVs(triW, texIndices.z, N, uv2X, uv2Y, uv2Z) : vec3(0.0);
            vec3 blended = tripNormal0 * w.x + tripNormal1 * w.y + tripNormal2 * w.z;
            worldNormal = normalize(blended);
            
            
            // Diagnostic: detect invalid/degenerate normals and show red so we can find broken pixels
            if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
                outColor = vec4(1.0, 0.0, 0.0, 1.0);
                return;
            }
        }
    } else {
        // Sample albedo from each layer and blend by barycentric weights
        vec3 a0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
    }

    // Compute normal mapping if enabled (per-material or global toggle)
    if (!usedTriplanar && ((materials[texIndices.x].mappingParams.x * w.x + materials[texIndices.y].mappingParams.x * w.y + materials[texIndices.z].mappingParams.x * w.z) > 0.5 || ubo.materialFlags.w > 0.5)) {
        // Sample normal map per-layer and blend in tangent space
        vec3 n0 = texture(normalArray, vec3(uv, float(texIndices.x))).rgb * 2.0 - 1.0;
        vec3 n1 = texture(normalArray, vec3(uv, float(texIndices.y))).rgb * 2.0 - 1.0;
        vec3 n2 = texture(normalArray, vec3(uv, float(texIndices.z))).rgb * 2.0 - 1.0;
        vec3 nmap = normalize(n0 * w.x + n1 * w.y + n2 * w.z);
        // Build TBN matrix from geometry for UV-space normal mapping
        vec3 T = normalize(dFdx(fragPosWorld));
        vec3 B = normalize(cross(N, T));
        T = normalize(cross(B, N)); // re-orthogonalize
        mat3 TBN = mat3(T, B, N);
        worldNormal = normalize(TBN * nmap);
        if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    // Sample roughness map (R channel)
    float roughnessValue;
    if (usedTriplanar) {
        float r0 = w.x > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float r1 = w.y > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float r2 = w.z > 0.0 ? computeTriplanarRoughnessUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    } else {
        float r0 = texture(roughnessArray, vec3(uv, float(texIndices.x))).r;
        float r1 = texture(roughnessArray, vec3(uv, float(texIndices.y))).r;
        float r2 = texture(roughnessArray, vec3(uv, float(texIndices.z))).r;
        roughnessValue = clamp(r0 * w.x + r1 * w.y + r2 * w.z, 0.0, 1.0);
    }
    if (!roughnessEnabled) roughnessValue = 0.0;

    // Sample ambient occlusion map (R channel)
    float ambientOcclusion;
    if (usedTriplanar) {
        float ao0 = w.x > 0.0 ? computeTriplanarAOUVs(triW, texIndices.x, uv0X, uv0Y, uv0Z) : 0.0;
        float ao1 = w.y > 0.0 ? computeTriplanarAOUVs(triW, texIndices.y, uv1X, uv1Y, uv1Z) : 0.0;
        float ao2 = w.z > 0.0 ? computeTriplanarAOUVs(triW, texIndices.z, uv2X, uv2Y, uv2Z) : 0.0;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    } else {
        float ao0 = texture(aoArray, vec3(uv, float(texIndices.x))).r;
        float ao1 = texture(aoArray, vec3(uv, float(texIndices.y))).r;
        float ao2 = texture(aoArray, vec3(uv, float(texIndices.z))).r;
        ambientOcclusion = clamp(ao0 * w.x + ao1 * w.y + ao2 * w.z, 0.0, 1.0);
    }

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation (skipped for brush pass to avoid self-shadowing)
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    float shadow = 0.0;
    bool isBrushPass = ubo.brushParams.z > 0.5;
    if (!isBrushPass && ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, fragPosWorld, bias);
        } else {
            shadow = 1.0;
        }
    }
    float totalShadow = shadow;

    // Blend material parameters (ambient/specular) by barycentric weights
    vec4 matFlags0 = materials[texIndices.x].materialFlags;
    vec4 matFlags1 = materials[texIndices.y].materialFlags;
    vec4 matFlags2 = materials[texIndices.z].materialFlags;
    vec4 blendedMatFlags = matFlags0 * w.x + matFlags1 * w.y + matFlags2 * w.z;
    // Ambient occlusion: sample texture value, blend with useAO flag and aoFactor
    // Blend roughnessAOParams across materials
    vec4 ra0 = materials[texIndices.x].roughnessAOParams;
    vec4 ra1 = materials[texIndices.y].roughnessAOParams;
    vec4 ra2 = materials[texIndices.z].roughnessAOParams;
    vec4 blendedRA = ra0 * w.x + ra1 * w.y + ra2 * w.z;
    
    float useAOf = blendedRA.z;
    float aoFactor = blendedRA.y;
    float roughnessFactor = blendedRA.x;
    
    float aoBlend = (useAOf > 0.5 && aoEnabled) ? ambientOcclusion : 1.0;
    aoBlend = mix(1.0, aoBlend, aoFactor);
    vec3 ambient = albedoColor * blendedMatFlags.z * aoBlend;
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);

    // Specular
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    vec4 spec0 = materials[texIndices.x].specularParams;
    vec4 spec1 = materials[texIndices.y].specularParams;
    vec4 spec2 = materials[texIndices.z].specularParams;
    vec4 blendedSpec = spec0 * w.x + spec1 * w.y + spec2 * w.z;
    // Gate specular on NdotL to prevent non-physical specular on dark-side surfaces
    // (normal-mapped worldNormal can make NdotL == 0 while still reflecting toward viewer).
    // Clamp shininess to >= 1.0 so pow(x, 0) == 1 never fires.
    float shininess = max(blendedSpec.y, 1.0);
    // Roughness modulates specular exponent: 0 = smooth (glossy), 1 = rough (diffuse)
    float specPower = mix(shininess, 1.0, roughnessValue * roughnessFactor);
    specPower = max(specPower, 1.0);
    float spec = (NdotL > 0.0) ? pow(max(dot(viewDir, reflectDir), 0.0), specPower) : 0.0;
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * blendedSpec.x;

    // Environment reflection (360° cubemap) — skipped during cubemap capture
    // (ubo.materialFlags.x is set to 1.0 by the 360 async task to avoid feedback).
    vec3 envReflection = vec3(0.0);
    float blendedRefStrength = 0.0;
    if (ubo.materialFlags.x < 0.5) {
        float refStrength0 = materials[texIndices.x].tessLevelParams.z;
        float refStrength1 = materials[texIndices.y].tessLevelParams.z;
        float refStrength2 = materials[texIndices.z].tessLevelParams.z;
        blendedRefStrength = refStrength0 * w.x + refStrength1 * w.y + refStrength2 * w.z;
        // Skip the cubemap fetch on non-reflective surfaces (the result is
        // multiplied by zero and mix() later collapses to the lit colour).
        if (blendedRefStrength > 1e-4) {
            vec3 envReflectDir = reflect(viewDir, worldNormal);
            vec3 envColor = texture(environmentMap, normalize(envReflectDir)).rgb;
            // Full-strength environment reflection at all angles
            float envBlend = 1.0;
            envReflection = envColor * envBlend * blendedRefStrength;
            // Apply global AO and roughness to environment reflection
            envReflection *= aoBlend * (1.0 - roughnessValue * roughnessFactor);
        }
    }

    // Debug visualisation modes (0 = normal render)
    int debugMode = int(ubo.debugParams.x + 0.5);
    if (debugMode == 1) {
        vec3 gn = normalize(fragNormal);
        outColor = vec4(gn * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 2) {
        vec3 nm = normalize(worldNormal);
        outColor = vec4(nm * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 3) {
        outColor = vec4(uv.x, uv.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 4) {
        outColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 5) {
        vec3 ra0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 ra1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 ra2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        vec3 rawAlbedo = ra0 * w.x + ra1 * w.y + ra2 * w.z;
        outColor = vec4(rawAlbedo, 1.0);
        return;
    }
    if (debugMode == 6) {
        vec3 rn0 = texture(normalArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 rn1 = texture(normalArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 rn2 = texture(normalArray, vec3(uv, float(texIndices.z))).rgb;
        vec3 rawNormalTex = rn0 * w.x + rn1 * w.y + rn2 * w.z;
        outColor = vec4(rawNormalTex, 1.0);
        return;
    }
    if (debugMode == 7) {
        float h0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h = h0 * w.x + h1 * w.y + h2 * w.z;
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == 8) {
        outColor = vec4(NdotL, totalShadow, 0.0, 1.0);
        return;
    }
    if (debugMode == 9) {
        vec3 normalToShow = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(normalToShow * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 10) {
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 11) {
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == 12) {
        outColor = vec4(shadow, 0.0, totalShadow, 1.0);
        return;
    }
    if (debugMode == 13) {
        // Visualize triplanar blend weights RGB (X/Y/Z projections)
        outColor = vec4(triW, 1.0);
        return;
    }

    if (debugMode == 14) {
        // Map each corner brushIndex to a distinct color from a small palette, then blend by barycentric weights
        const int PALETTE_SIZE = 16;
        const vec3 palette[PALETTE_SIZE] = vec3[](
            vec3(0.90, 0.10, 0.10), // red
            vec3(0.10, 0.90, 0.10), // green
            vec3(0.10, 0.10, 0.90), // blue
            vec3(0.90, 0.90, 0.10), // yellow
            vec3(0.90, 0.10, 0.90), // magenta
            vec3(0.10, 0.90, 0.90), // cyan
            vec3(1.00, 0.55, 0.10), // orange
            vec3(0.55, 0.35, 0.15), // brown
            vec3(0.60, 0.20, 0.80), // purple
            vec3(1.00, 0.50, 0.70), // pink
            vec3(0.70, 1.00, 0.30), // lime
            vec3(0.00, 0.45, 0.55), // teal
            vec3(0.05, 0.10, 0.35), // navy
            vec3(0.45, 0.50, 0.10), // olive
            vec3(0.60, 0.60, 0.60), // gray
            vec3(1.00, 1.00, 1.00)  // white
        );

        vec3 c0 = palette[int(mod(float(texIndices.x), float(PALETTE_SIZE)) + 0.5)];
        vec3 c1 = palette[int(mod(float(texIndices.y), float(PALETTE_SIZE)) + 0.5)];
        vec3 c2 = palette[int(mod(float(texIndices.z), float(PALETTE_SIZE)) + 0.5)];
        vec3 blended = c0 * w.x + c1 * w.y + c2 * w.z;
        outColor = vec4(blended, 1.0);
        return;
    }

    if (debugMode == 15) {
        // Visualize barycentric weights directly as RGB
        outColor = vec4(clamp(w, 0.0, 1.0), 1.0);
        return;
    }

    if (debugMode == 16) {
        // Show the raw albedo samples for each corner packed into RGB (a0.r, a1.r, a2.r)
        vec3 a0 = texture(albedoArray, vec3(uv, float(texIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(texIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(texIndices.z))).rgb;
        outColor = vec4(a0.r, a1.r, a2.r, 1.0);
        return;
    }

    if (debugMode == 17) {
        // Visualize triplanar-sampled albedo blended across the three material indices
        vec3 ta0 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.x, N);
        vec3 ta1 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.y, N);
        vec3 ta2 = computeTriplanarAlbedo(fragPosWorld, triW, texIndices.z, N);
        vec3 tAlbedo = ta0 * w.x + ta1 * w.y + ta2 * w.z;
        outColor = vec4(tAlbedo, 1.0);
        return;
    }

    if (debugMode == 18) {
        // Show per-projection triplanar heights for each corner packed into RGB
        vec2 tScale = vec2(materials[texIndices.x].triplanarParams.x, 
                            materials[texIndices.x].triplanarParams.y);
        float th0x = texture(heightArray, vec3(fragPosWorld.yz * tScale, float(texIndices.x))).r;
        float th0y = texture(heightArray, vec3(fragPosWorld.xz * tScale, float(texIndices.x))).r;
        float th0z = texture(heightArray, vec3(fragPosWorld.xy * tScale, float(texIndices.x))).r;
        // Pack the three projection samples as RGB for the first material (useful to see which projection contributes height)
        outColor = vec4(th0x, th0y, th0z, 1.0);
        return;
    }

    if (debugMode == 19) {
        // Show difference between UV-blended height and triplanar-blended height (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, geomN, texIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 20) {
        // Visualize triplanar-sampled normal blended across the three material indices
        vec3 tn0 = computeTriplanarNormal(fragPosWorld, triW, texIndices.x, geomN, N);
        vec3 tn1 = computeTriplanarNormal(fragPosWorld, triW, texIndices.y, geomN, N);
        vec3 tn2 = computeTriplanarNormal(fragPosWorld, triW, texIndices.z, geomN, N);
        vec3 blended = tn0 * w.x + tn1 * w.y + tn2 * w.z;
        vec3 tNormal = reorientNormal(blended, geomN);
        outColor = vec4(tNormal * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 21) {
        // Show per-projection triplanar normals for the first material packed into RGB
        vec3 nX = computeTriplanarNormal(fragPosWorld, vec3(1.0, 0.0, 0.0), texIndices.x, geomN, N);
        vec3 nY = computeTriplanarNormal(fragPosWorld, vec3(0.0, 1.0, 0.0), texIndices.x, geomN, N);
        vec3 nZ = computeTriplanarNormal(fragPosWorld, vec3(0.0, 0.0, 1.0), texIndices.x, geomN, N);
        // Pack single components of each projection to RGB so we can visually inspect contributions
        outColor = vec4(nX.x * 0.5 + 0.5, nY.y * 0.5 + 0.5, nZ.z * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 22) {
        // Visualize triplanar-sampled bump (height) blended across the three material indices
        float b0 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x);
        float b1 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y);
        float b2 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z);
        float b = b0 * w.x + b1 * w.y + b2 * w.z;
        outColor = vec4(vec3(b), 1.0);
        return;
    }

    if (debugMode == 23) {
        // Show per-projection triplanar heights using sampleHeightTriplanar for the first material packed into RGB
        float ph0x = sampleHeightTriplanar(fragPosWorld, vec3(1.0, 0.0, 0.0), texIndices.x);
        float ph0y = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 1.0, 0.0), texIndices.x);
        float ph0z = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 0.0, 1.0), texIndices.x);
        outColor = vec4(ph0x, ph0y, ph0z, 1.0);
        return;
    }

    if (debugMode == 24) {
        // Show difference between UV-blended height and triplanar-blended height using worldNormal (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(texIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(texIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(texIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, worldNormal, texIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 25) {
        // Visualize triplanar UV for X projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvX);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 26) {
        // Visualize triplanar UV for Y projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvY);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 27) {
        // Visualize triplanar UV for Z projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, texIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvZ);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }

    if (debugMode == 28) {
        tripNormal0 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.x, geomN, N);
        outColor = vec4(normalize(tripNormal0) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 29) {
        tripNormal1 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.y, geomN, N);
        outColor = vec4(normalize(tripNormal1) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 30) {
        tripNormal2 = computeTriplanarNormal(fragPosWorldNotDisplaced, triW, texIndices.z, geomN, N);
        outColor = vec4(normalize(tripNormal2) * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 31) {
        // Visualize TES-provided face normal (sharp per-triangle normal computed in tessellation evaluation shader)
        vec3 s = normalize(fragSharpNormal);
        outColor = vec4(s * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 32) {
        outColor = vec4(vec3(roughnessValue), 1.0);
        return;
    }
    if (debugMode == 33) {
        outColor = vec4(vec3(ambientOcclusion), 1.0);
        return;
    }
    if (debugMode == 49) {
        // Environment reflection factor — shows the cubemap sample shaded by
        // Fresnel and per-material reflectionStrength (pre-multiplied).
        outColor = vec4(envReflection, 1.0);
        return;
    }

    // Blend between lit color and environment reflection.
    // reflectionStrength=0 → lit color only, =1 → pure mirror.
    vec3 finalColor = mix(ambient + diffuse + specular, envReflection, blendedRefStrength);

    // REMOVE mode: smoothly fade between red tint and brush texture at 1Hz
    if (brushRedFade > 0.0) {
        vec3 redTint = vec3(finalColor.r, 0.0, 0.0);
        finalColor = mix(redTint, finalColor, brushRedFade);
    }
    
    // DEBUG: Visualize lighting components
    // Uncomment to debug:
    // if (length(albedoColor) < 0.01) { outColor = vec4(1.0, 0.0, 0.0, 1.0); return; } // Red if no albedo
    // if (length(finalColor) < 0.01) { outColor = vec4(0.0, 1.0, 0.0, 1.0); return; } // Green if no lighting
    // outColor = vec4(albedoColor, 1.0); return; // Show raw albedo
    // outColor = vec4(vec3(NdotL), 1.0); return; // Show N·L term
    
    // Final output (single color target)
    outColor = vec4(finalColor, 1.0);
}