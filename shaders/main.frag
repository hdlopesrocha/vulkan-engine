#version 450


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragNormal;
layout(location = 5) flat in ivec3 fragTexIndices;
layout(location = 11) in vec3 fragTexWeights;
layout(location = 4) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;
layout(location = 10) in vec3 fragSharpNormal; // face normal computed in TES (sharp)
layout(location = 7) in vec3 fragPosWorldNotDisplaced;

#include "includes/ubo.glsl"

#include "includes/textures.glsl"

layout(location = 0) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"

#include "includes/shadows.glsl"

void main() {
    bool isShadowPass = ubo.passParams.x > 0.5;

    // Fast-path for shadow pass: skip expensive lighting/texture work.
    if (isShadowPass) {
        outColor = vec4(0.0);
        return;
    }
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
    
    // Compute fragment tangent aligned with the dominant triplanar axis.
    // This mirrors the CPU-side triplanar conventions so T/B are stable.
    vec3 tAxis;
    float ax = abs(geomN.x), ay = abs(geomN.y), az = abs(geomN.z);
    if (ax >= ay && ax >= az) {
        float s = (geomN.x >= 0.0) ? 1.0 : -1.0;
        vec3 axisX = vec3(-s, 0.0, 0.0);
        vec3 uDirX = vec3(0.0, 0.0, -s);
        tAxis = normalize(uDirX - axisX * dot(axisX, uDirX));
    } else if (ay >= ax && ay >= az) {
        float s = (geomN.y >= 0.0) ? 1.0 : -1.0;
        vec3 axisY = vec3(0.0, -s, 0.0);
        vec3 uDirY = vec3(1.0, 0.0, 0.0);
        tAxis = normalize(uDirY - axisY * dot(axisY, uDirY));
    } else {
        float s = (geomN.z >= 0.0) ? 1.0 : -1.0;
        vec3 axisZ = vec3(0.0, 0.0, -s);
        vec3 uDirZ = vec3(s, 0.0, 0.0);
        tAxis = normalize(uDirZ - axisZ * dot(axisZ, uDirZ));
    }
    vec3 tang = normalize(tAxis - N * dot(N, tAxis));
    float handed = 1.0;
    vec4 fragTangent = vec4(tang, handed);
    vec3 T = normalize(fragTangent.xyz);
    vec3 B = cross(N, T) * fragTangent.w;
    bool haveTB = true;

    // Sample albedo texture (triplanar when enabled)
    vec3 albedoColor;

    // Mix triplanar flag across the three materials
    float triFlag = dot(vec3(materials[fragTexIndices.x].triplanarParams.z, materials[fragTexIndices.y].triplanarParams.z, materials[fragTexIndices.z].triplanarParams.z), w);
    if (triFlag > 0.5) {
        usedTriplanar = true;
        // compute triplanar albedo per-layer then blend
        vec3 a0 = w.x > 0.0 ? computeTriplanarAlbedo(fragPosWorldNotDisplaced, triW, fragTexIndices.x, geomN) : vec3(0.0);
        vec3 a1 = w.y > 0.0 ? computeTriplanarAlbedo(fragPosWorldNotDisplaced, triW, fragTexIndices.y, geomN) : vec3(0.0);
        vec3 a2 = w.z > 0.0 ? computeTriplanarAlbedo(fragPosWorldNotDisplaced, triW, fragTexIndices.z, geomN) : vec3(0.0);
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
        // If normal mapping/triplanar normal enabled per-material or global, compute blended triplanar normal
        float mapFlag0 = materials[fragTexIndices.x].mappingParams.x;
        float mapFlag1 = materials[fragTexIndices.y].mappingParams.x;
        float mapFlag2 = materials[fragTexIndices.z].mappingParams.x;
        if ((mapFlag0 * w.x + mapFlag1 * w.y + mapFlag2 * w.z) > 0.5 || ubo.materialFlags.w > 0.5) {
            vec3 n0 = w.x > 0.0 ? computeTriplanarNormal(fragPosWorldNotDisplaced, triW, fragTexIndices.x, geomN, fragTangent) : vec3(0.0);
            vec3 n1 = w.y > 0.0 ? computeTriplanarNormal(fragPosWorldNotDisplaced, triW, fragTexIndices.y, geomN, fragTangent) : vec3(0.0);
            vec3 n2 = w.z > 0.0 ? computeTriplanarNormal(fragPosWorldNotDisplaced, triW, fragTexIndices.z, geomN, fragTangent) : vec3(0.0);
            worldNormal = normalize(n0 * w.x + n1 * w.y + n2 * w.z);
            // Diagnostic: detect invalid/degenerate normals and show red so we can find broken pixels
            if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
                outColor = vec4(1.0, 0.0, 0.0, 1.0);
                return;
            }
        }
    } else {
        // Sample albedo from each layer and blend by barycentric weights
        vec3 a0 = texture(albedoArray, vec3(uv, float(fragTexIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(fragTexIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(fragTexIndices.z))).rgb;
        albedoColor = a0 * w.x + a1 * w.y + a2 * w.z;
    }

    // Compute normal mapping if enabled (per-material or global toggle)
    if (!usedTriplanar && ((materials[fragTexIndices.x].mappingParams.x * w.x + materials[fragTexIndices.y].mappingParams.x * w.y + materials[fragTexIndices.z].mappingParams.x * w.z) > 0.5 || ubo.materialFlags.w > 0.5)) {
        // Sample normal map per-layer and blend in tangent space
        vec3 n0 = texture(normalArray, vec3(uv, float(fragTexIndices.x))).rgb * 2.0 - 1.0;
        vec3 n1 = texture(normalArray, vec3(uv, float(fragTexIndices.y))).rgb * 2.0 - 1.0;
        vec3 n2 = texture(normalArray, vec3(uv, float(fragTexIndices.z))).rgb * 2.0 - 1.0;
        vec3 nmap = normalize(n0 * w.x + n1 * w.y + n2 * w.z);
        // Blend per-material normal convention flags by barycentric weights
        vec4 blendedNormalParams = materials[fragTexIndices.x].normalParams * w.x + materials[fragTexIndices.y].normalParams * w.y + materials[fragTexIndices.z].normalParams * w.z;
        vec3 tmpT, tmpB;
        vec3 computedWorld;
        if (computeWorldNormalFromNormalMap(fragTangent, fragPosWorld, uv, N, nmap, blendedNormalParams, computedWorld, tmpT, tmpB)) {
            worldNormal = normalize(computedWorld);
            // update T/B so subsequent code that reads T/B remains consistent
            T = normalize(tmpT);
            B = normalize(tmpB);
        } else {
            // Fallback: use projection TBN
            mat3 tbn = mat3(T, B, N);
            worldNormal = normalize(tbn * nmap);
        }
        if (isnan(worldNormal.x) || isnan(worldNormal.y) || isnan(worldNormal.z) || length(worldNormal) < 1e-6) {
            outColor = vec4(1.0, 0.0, 0.0, 1.0);
            return;
        }
    }

    // Lighting calculation
    vec3 toLight = -normalize(ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, toLight), 0.0);

    // Shadow calculation
    vec4 adjustedPosLightSpace = fragPosLightSpace;
    float shadow = 0.0;
    if (ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.002 * (1.0 - NdotL), 0.0005);
            shadow = ShadowCalculation(adjustedPosLightSpace, bias);
        } else {
            shadow = 1.0;
        }
    }
    float totalShadow = shadow;

    // Blend material parameters (ambient/specular) by barycentric weights
    vec4 matFlags0 = materials[fragTexIndices.x].materialFlags;
    vec4 matFlags1 = materials[fragTexIndices.y].materialFlags;
    vec4 matFlags2 = materials[fragTexIndices.z].materialFlags;
    vec4 blendedMatFlags = matFlags0 * w.x + matFlags1 * w.y + matFlags2 * w.z;
    vec3 ambient = albedoColor * blendedMatFlags.z;
    vec3 diffuse = albedoColor * ubo.lightColor.rgb * NdotL * (1.0 - totalShadow);

    // Specular
    vec3 viewDir = normalize(ubo.viewPos.xyz - fragPosWorld);
    vec3 reflectDir = reflect(-toLight, worldNormal);
    vec4 spec0 = materials[fragTexIndices.x].specularParams;
    vec4 spec1 = materials[fragTexIndices.y].specularParams;
    vec4 spec2 = materials[fragTexIndices.z].specularParams;
    vec4 blendedSpec = spec0 * w.x + spec1 * w.y + spec2 * w.z;
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), blendedSpec.y);
    vec3 specular = ubo.lightColor.rgb * spec * (1.0 - totalShadow) * blendedSpec.x;

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
        outColor = vec4(T * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 5) {
        outColor = vec4(B * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 6) {
        outColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 7) {
        vec3 ra0 = texture(albedoArray, vec3(uv, float(fragTexIndices.x))).rgb;
        vec3 ra1 = texture(albedoArray, vec3(uv, float(fragTexIndices.y))).rgb;
        vec3 ra2 = texture(albedoArray, vec3(uv, float(fragTexIndices.z))).rgb;
        vec3 rawAlbedo = ra0 * w.x + ra1 * w.y + ra2 * w.z;
        outColor = vec4(rawAlbedo, 1.0);
        return;
    }
    if (debugMode == 8) {
        vec3 rn0 = texture(normalArray, vec3(uv, float(fragTexIndices.x))).rgb;
        vec3 rn1 = texture(normalArray, vec3(uv, float(fragTexIndices.y))).rgb;
        vec3 rn2 = texture(normalArray, vec3(uv, float(fragTexIndices.z))).rgb;
        vec3 rawNormalTex = rn0 * w.x + rn1 * w.y + rn2 * w.z;
        outColor = vec4(rawNormalTex, 1.0);
        return;
    }
    if (debugMode == 9) {
        float h0 = texture(heightArray, vec3(uv, float(fragTexIndices.x))).r;
        float h1 = texture(heightArray, vec3(uv, float(fragTexIndices.y))).r;
        float h2 = texture(heightArray, vec3(uv, float(fragTexIndices.z))).r;
        float h = h0 * w.x + h1 * w.y + h2 * w.z;
        outColor = vec4(vec3(h), 1.0);
        return;
    }
    if (debugMode == 10) {
        outColor = vec4(NdotL, totalShadow, 0.0, 1.0);
        return;
    }
    if (debugMode == 11) {
        vec3 normalToShow = normalize(cross(dFdy(fragPosWorld), dFdx(fragPosWorld)));
        outColor = vec4(normalToShow * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 12) {
        vec3 tl = normalize(toLight);
        outColor = vec4(tl * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 13) {
        outColor = vec4(vec3(NdotL), 1.0);
        return;
    }
    if (debugMode == 14) {
        outColor = vec4(shadow, 0.0, totalShadow, 1.0);
        return;
    }
    if (debugMode == 15) {
        // Visualize triplanar blend weights RGB (X/Y/Z projections)
        vec3 w = triW;
        outColor = vec4(w, 1.0);
        return;
    }

    if (debugMode == 16) {
        // Map each corner texIndex to a distinct color from a small palette, then blend by barycentric weights
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

        vec3 c0 = palette[int(mod(float(fragTexIndices.x), float(PALETTE_SIZE)) + 0.5)];
        vec3 c1 = palette[int(mod(float(fragTexIndices.y), float(PALETTE_SIZE)) + 0.5)];
        vec3 c2 = palette[int(mod(float(fragTexIndices.z), float(PALETTE_SIZE)) + 0.5)];
        vec3 blended = c0 * w.x + c1 * w.y + c2 * w.z;
        outColor = vec4(blended, 1.0);
        return;
    }

    if (debugMode == 17) {
        // Visualize barycentric weights directly as RGB
        outColor = vec4(clamp(w, 0.0, 1.0), 1.0);
        return;
    }

    if (debugMode == 18) {
        // Show the raw albedo samples for each corner packed into RGB (a0.r, a1.r, a2.r)
        vec3 a0 = texture(albedoArray, vec3(uv, float(fragTexIndices.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(fragTexIndices.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(fragTexIndices.z))).rgb;
        outColor = vec4(a0.r, a1.r, a2.r, 1.0);
        return;
    }

    if (debugMode == 19) {
        // Visualize triplanar-sampled albedo blended across the three material indices
        vec3 ta0 = computeTriplanarAlbedo(fragPosWorld, triW, fragTexIndices.x, N);
        vec3 ta1 = computeTriplanarAlbedo(fragPosWorld, triW, fragTexIndices.y, N);
        vec3 ta2 = computeTriplanarAlbedo(fragPosWorld, triW, fragTexIndices.z, N);
        vec3 tAlbedo = ta0 * w.x + ta1 * w.y + ta2 * w.z;
        outColor = vec4(tAlbedo, 1.0);
        return;
    }

    if (debugMode == 20) {
        // Show per-projection triplanar heights for each corner packed into RGB
        float th0x = texture(heightArray, vec3(fragPosWorld.yz * vec2(materials[fragTexIndices.x].triplanarParams.x, materials[fragTexIndices.x].triplanarParams.y), float(fragTexIndices.x))).r;
        float th0y = texture(heightArray, vec3(fragPosWorld.xz * vec2(materials[fragTexIndices.x].triplanarParams.x, materials[fragTexIndices.x].triplanarParams.y), float(fragTexIndices.x))).r;
        float th0z = texture(heightArray, vec3(fragPosWorld.xy * vec2(materials[fragTexIndices.x].triplanarParams.x, materials[fragTexIndices.x].triplanarParams.y), float(fragTexIndices.x))).r;
        // Pack the three projection samples as RGB for the first material (useful to see which projection contributes height)
        outColor = vec4(th0x, th0y, th0z, 1.0);
        return;
    }

    if (debugMode == 21) {
        // Show difference between UV-blended height and triplanar-blended height (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(fragTexIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(fragTexIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(fragTexIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, geomN, fragTexIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, geomN, fragTexIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, geomN, fragTexIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 22) {
        // Visualize triplanar-sampled normal blended across the three material indices
        vec3 tn0 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.x, geomN, fragTangent);
        vec3 tn1 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.y, geomN, fragTangent);
        vec3 tn2 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.z, geomN, fragTangent);
        vec3 tNormal = normalize(tn0 * w.x + tn1 * w.y + tn2 * w.z);
        outColor = vec4(tNormal * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 23) {
        // Show per-projection triplanar normals for the first material packed into RGB
        vec3 nX = computeTriplanarNormal(fragPosWorld, vec3(1.0, 0.0, 0.0), fragTexIndices.x, geomN, fragTangent);
        vec3 nY = computeTriplanarNormal(fragPosWorld, vec3(0.0, 1.0, 0.0), fragTexIndices.x, geomN, fragTangent);
        vec3 nZ = computeTriplanarNormal(fragPosWorld, vec3(0.0, 0.0, 1.0), fragTexIndices.x, geomN, fragTangent);
        // Pack single components of each projection to RGB so we can visually inspect contributions
        outColor = vec4(nX.x * 0.5 + 0.5, nY.y * 0.5 + 0.5, nZ.z * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 24) {
        // Show difference between UV-sampled normal (converted to world-space) and triplanar-blended normal
        vec3 nu0 = texture(normalArray, vec3(uv, float(fragTexIndices.x))).rgb * 2.0 - 1.0;
        vec3 nu1 = texture(normalArray, vec3(uv, float(fragTexIndices.y))).rgb * 2.0 - 1.0;
        vec3 nu2 = texture(normalArray, vec3(uv, float(fragTexIndices.z))).rgb * 2.0 - 1.0;
        vec3 nmap = normalize(nu0 * w.x + nu1 * w.y + nu2 * w.z);
        vec3 uvWorld;
        vec3 tmpT, tmpB;
        // Blend per-material normal convention flags by barycentric weights
        vec4 blendedNormalParams = materials[fragTexIndices.x].normalParams * w.x + materials[fragTexIndices.y].normalParams * w.y + materials[fragTexIndices.z].normalParams * w.z;
        if (!computeWorldNormalFromNormalMap(fragTangent, fragPosWorld, uv, N, nmap, blendedNormalParams, uvWorld, tmpT, tmpB)) {
            uvWorld = N; // fallback
        }
        vec3 tn0 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.x, geomN, fragTangent);
        vec3 tn1 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.y, geomN, fragTangent);
        vec3 tn2 = computeTriplanarNormal(fragPosWorld, triW, fragTexIndices.z, geomN, fragTangent);
        vec3 triWorld = normalize(tn0 * w.x + tn1 * w.y + tn2 * w.z);
        float angle = acos(clamp(dot(normalize(uvWorld), triWorld), -1.0, 1.0));
        float d = angle / 3.14159265; // normalize to 0..1
        outColor = vec4(vec3(d * 5.0), 1.0);
        return;
    }

    if (debugMode == 25) {
        // Visualize triplanar-sampled bump (height) blended across the three material indices
        float b0 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.x);
        float b1 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.y);
        float b2 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.z);
        float b = b0 * w.x + b1 * w.y + b2 * w.z;
        outColor = vec4(vec3(b), 1.0);
        return;
    }

    if (debugMode == 26) {
        // Show per-projection triplanar heights using sampleHeightTriplanar for the first material packed into RGB
        float ph0x = sampleHeightTriplanar(fragPosWorld, vec3(1.0, 0.0, 0.0), fragTexIndices.x);
        float ph0y = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 1.0, 0.0), fragTexIndices.x);
        float ph0z = sampleHeightTriplanar(fragPosWorld, vec3(0.0, 0.0, 1.0), fragTexIndices.x);
        outColor = vec4(ph0x, ph0y, ph0z, 1.0);
        return;
    }

    if (debugMode == 27) {
        // Show difference between UV-blended height and triplanar-blended height using worldNormal (abs difference)
        float h_uv0 = texture(heightArray, vec3(uv, float(fragTexIndices.x))).r;
        float h_uv1 = texture(heightArray, vec3(uv, float(fragTexIndices.y))).r;
        float h_uv2 = texture(heightArray, vec3(uv, float(fragTexIndices.z))).r;
        float h_uv = h_uv0 * w.x + h_uv1 * w.y + h_uv2 * w.z;
        float h_tri0 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.x);
        float h_tri1 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.y);
        float h_tri2 = sampleHeightTriplanar(fragPosWorld, worldNormal, fragTexIndices.z);
        float h_tri = h_tri0 * w.x + h_tri1 * w.y + h_tri2 * w.z;
        float d = abs(h_uv - h_tri);
        outColor = vec4(vec3(d * 5.0), 1.0); // amplify differences for visibility
        return;
    }

    if (debugMode == 28) {
        // Visualize triplanar UV for X projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, fragTexIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvX);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 29) {
        // Visualize triplanar UV for Y projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, fragTexIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvY);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }
    if (debugMode == 30) {
        // Visualize triplanar UV for Z projection (first material)
        vec2 uvX, uvY, uvZ;
        computeTriplanarUVs(fragPosWorld, fragTexIndices.x, N, uvX, uvY, uvZ);
        vec2 show = fract(uvZ);
        outColor = vec4(show.x, show.y, 0.0, 1.0);
        return;
    }

    if (debugMode == 31) {
        // Visualize per-projection X triplanar normal (first material)
        vec3 nX = computeTriplanarNormal(fragPosWorld, vec3(1.0, 0.0, 0.0), fragTexIndices.x, N, fragTangent);
        outColor = vec4(normalize(nX) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 32) {
        // Visualize per-projection Y triplanar normal (first material)
        vec3 nY = computeTriplanarNormal(fragPosWorld, vec3(0.0, 1.0, 0.0), fragTexIndices.x, N, fragTangent);
        outColor = vec4(normalize(nY) * 0.5 + 0.5, 1.0);
        return;
    }
    if (debugMode == 33) {
        // Visualize per-projection Z triplanar normal (first material)
        vec3 nZ = computeTriplanarNormal(fragPosWorld, vec3(0.0, 0.0, 1.0), fragTexIndices.x, N, fragTangent);
        outColor = vec4(normalize(nZ) * 0.5 + 0.5, 1.0);
        return;
    }

    if (debugMode == 34) {
        // Visualize TES-provided face normal (sharp per-triangle normal computed in tessellation evaluation shader)
        vec3 s = normalize(fragSharpNormal);
        outColor = vec4(s * 0.5 + 0.5, 1.0);
        return;
    }

    // Fallback to normal rendering
    outColor = vec4(ambient + diffuse + specular, 1.0);
}