// Water TES (moved from water.tese). Requires: ubo, locations,
// perlin, water_noise. Defines the stage's main().


// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, cw) in;

layout(location = VARY_LOCALPOS) in vec3 inPos[];
layout(location = VARY_NORMAL) in vec3 inNormal[];
layout(location = VARY_SHARPNORMAL) in vec3 inBaseNormal[];
layout(location = VARY_UV) in vec2 inTexCoord[];
layout(location = VARY_BRUSHPATCH) in ivec3 tc_fragBrushIndex[];
layout(location = VARY_TEXWEIGHTS) in vec3 tc_fragTexWeights[];
layout(location = VARY_HSV) in vec3 tc_fragHSV[];

layout(location = VARY_LOCALPOS) out vec3 fragPos;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) out vec3 fragBaseNormal;  // undisplaced base normal for per-fragment detail
layout(location = VARY_BASEPOS) out vec4 fragBasePos;        // xyz = undisplaced base position, w = raw bump amplitude
layout(location = VARY_WATERDEPTH) out float fragWaterDepth; // measured water depth (-1 = unknown/deep)
layout(location = VARY_SHOREDIR) out vec2 fragShoreDir;      // unit shore direction (toward thinner water)
layout(location = VARY_UV) out vec2 fragTexCoord;
layout(location = VARY_POSCLIP) out vec4 fragPosClip;  // clip-space position for depth lookup
layout(location = VARY_DEBUG) out vec3 fragDebug;   // debug visual (displacement)
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;
layout(location = VARY_HSV) out vec3 fragHSV;


// Solid terrain depth (set 2, binding 5). The shore-wave zones are driven by
// the water depth measured against the REAL bottom here, not by the water
// volume's back face: the volume underside is pushed an SDF bias below the
// terrain (to keep the surface from z-fighting the ground), which shifts every
// zone by that bias. This binding is also used by the fragment stage, so the
// pass already depends on the solid depth target.
layout(set = 2, binding = 5) uniform sampler2D solidSceneDepthTex;
// Water volume back-face depth (set 2, binding 0): direction fallback only.
// Where no solid bottom stands behind the pixel (very deep / far water) the
// volume underside still carries the bottom slope, and its constant SDF bias
// does not change the horizontal gradient direction.
layout(set = 2, binding = 0) uniform sampler2D waterBackDepthTex;

// World-space direction of DECREASING water depth (toward the shore) from a
// center raw-depth sample plus two offset samples. Returns vec2(0) when the
// signal is unusable (clear depth, degenerate span, flat bottom).
vec2 waterShoreDirFromSamples(float rawC, float rawX, float rawY,
                              vec2 uvC, vec2 uvX, vec2 uvY, vec3 surfacePos) {
    if (rawC >= 0.9999 || rawX >= 0.9999 || rawY >= 0.9999) return vec2(0.0);
    vec4 wC = ubo.invViewProjection * vec4(uvC * 2.0 - 1.0, rawC, 1.0);
    vec4 wX = ubo.invViewProjection * vec4(uvX * 2.0 - 1.0, rawX, 1.0);
    vec4 wY = ubo.invViewProjection * vec4(uvY * 2.0 - 1.0, rawY, 1.0);
    vec3 pC = wC.xyz / wC.w;
    vec3 pX = wX.xyz / wX.w;
    vec3 pY = wY.xyz / wY.w;
    float dC = max(surfacePos.y - pC.y, 0.0);
    float dX = max(surfacePos.y - pX.y, 0.0);
    float dY = max(surfacePos.y - pY.y, 0.0);
    // Screen step -> world XZ offsets; solve the 2x2 system for the
    // world-space depth gradient, then take its negative (decreasing depth).
    vec2 sX = pX.xz - pC.xz;
    vec2 sY = pY.xz - pC.xz;
    float det = sX.x * sY.y - sX.y * sY.x;
    if (abs(det) < 1e-4) return vec2(0.0);
    vec2 g = vec2((dX - dC) * sY.y - (dY - dC) * sX.y,
                  sX.x * (dY - dC) - sY.x * (dX - dC)) / det;
    float gl = length(g);
    if (gl < 1e-5) return vec2(0.0);
    return -g / gl;
}

void main() {
    // Interpolate position
        vec3 bary = gl_TessCoord;

        // Interpolate position
        vec3 pos = bary.x * inPos[0] +
                   bary.y * inPos[1] +
                   bary.z * inPos[2];
    
    // Interpolate normal
        vec3 normal = normalize(bary.x * inNormal[0] +
                                bary.y * inNormal[1] +
                                bary.z * inNormal[2]);
    
    // Interpolate texture coordinates
        fragTexCoord = bary.x * inTexCoord[0] +
                       bary.y * inTexCoord[1] +
                       bary.z * inTexCoord[2];
    
    // Interpolate HSV
    fragHSV = tc_fragHSV[0] * bary.x + tc_fragHSV[1] * bary.y + tc_fragHSV[2] * bary.z;

    // Select per-patch brushIndex from compressed TCS outputs (tc_fragBrushIndex / tc_fragTexWeights)
    ivec3 texIndices = max(tc_fragBrushIndex[0], ivec3(0));
    vec3 weights = tc_fragTexWeights[0] * bary.x + tc_fragTexWeights[1] * bary.y + tc_fragTexWeights[2] * bary.z;
    int chosenIdx = texIndices.x;
    if (texIndices.y >= 0 && weights.y > weights.x) chosenIdx = texIndices.y;
    if (texIndices.z >= 0 && weights.z > max(weights.x, weights.y)) chosenIdx = texIndices.z;
    if (chosenIdx < 0) chosenIdx = 0;
    // Expose the chosen brushIndex to the fragment stage (kept raw so the
    // fragment debug view can show the true id distribution)
    fragBrushIndex = chosenIdx;

    // Load selected WaterParams from SSBO, falling back to layer 0 for
    // out-of-range terrain paint ids (see water.frag).
    int nWL = max(waterParams.length(), 1);
    WaterParamsGPU wp = waterParams[(chosenIdx >= 0 && chosenIdx < nWL) ? chosenIdx : 0];

    // Get the water parameters driving the single wave field.
    float time = waterRenderUBO.timeParams.x;
    float noiseTimeSpeed = wp.params3.x;

    float bumpAmp = wp.waveParams.z; // bump amplitude provided via Water widget

    // --- Screen-space UV of the undisplaced base vertex ---
    // Used to sample the water back-face depth (thickness) below.
    vec2 screenUV = vec2(0.0);
    bool haveScreen = false;
    {
        vec4 preClip = ubo.viewProjection * vec4(pos, 1.0);
        if (preClip.w > 0.001) {
            screenUV = clamp(preClip.xy / preClip.w * 0.5 + 0.5, 0.001, 0.999);
            haveScreen = true;
        }
    }

    // --- Water depth + shore direction (drive the shore-wave zones) ---
    // Depth is measured against the SOLID terrain behind the water (the
    // visible bottom), so depth = 0 at the true waterline and no SDF bias from
    // the water volume's underside leaks into the zones. The drop is vertical
    // (world Y difference along the view ray), which keeps the region bands
    // camera-stable. No solid bottom behind the pixel -> -1 (open/deep water
    // keeps the full swell).
    //
    // The shore direction is the direction of DECREASING water depth (the
    // negative water-depth gradient): waves and their foam travel toward
    // thinning water. The gradient is a central sample of the solid depth at
    // two screen offsets, reconstructed in world space. In deep water the
    // solid bottom can be too far/absent, so the water volume's own back face
    // (whose constant SDF bias is vertical and does not change the horizontal
    // gradient direction) provides the direction fallback. Finally the
    // configured shoreWaveAngle is used when neither measures a slope.
    vec2 shoreDir = normalize(wp.waveDirection.xy + vec2(1e-5, 0.0));
    float waterDepth = -1.0;
    if (haveScreen) {
        float solidDepthRaw = texture(solidSceneDepthTex, screenUV).r;
        if (solidDepthRaw < 0.9999) {
            vec4 solidWorldH = ubo.invViewProjection * vec4(screenUV * 2.0 - 1.0, solidDepthRaw, 1.0);
            waterDepth = max(pos.y - solidWorldH.y / solidWorldH.w, 0.0);
        }

        float gradStep = max(wp.waveWarp.w, 0.0);
        if (gradStep > 0.0) {
            vec2 texel = 1.0 / vec2(textureSize(solidSceneDepthTex, 0));
            vec2 uvX = clamp(screenUV + vec2(texel.x * gradStep, 0.0), 0.0, 1.0);
            vec2 uvY = clamp(screenUV + vec2(0.0, texel.y * gradStep), 0.0, 1.0);

            vec2 dir = waterShoreDirFromSamples(
                solidDepthRaw,
                texture(solidSceneDepthTex, uvX).r,
                texture(solidSceneDepthTex, uvY).r,
                screenUV, uvX, uvY, pos);
            if (dot(dir, dir) < 1e-6) {
                dir = waterShoreDirFromSamples(
                    texture(waterBackDepthTex, screenUV).r,
                    texture(waterBackDepthTex, uvX).r,
                    texture(waterBackDepthTex, uvY).r,
                    screenUV, uvX, uvY, pos);
            }
            if (dot(dir, dir) > 1e-6) shoreDir = dir;
        }
    }
    fragShoreDir = shoreDir;

    // Calculate the thickness-zoned shore-wave displacement and its analytic
    // spatial gradient (single wave field: directional swell + chop + mask).
    float animTime = time * noiseTimeSpeed;
    vec3 xyz = pos.xyz;

    // Surface basis (tangent plane) for projecting the analytic gradient.
    vec3 upVec = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(upVec, normal));
    vec3 B = cross(normal, T);

    vec4 wave = waterWaveSample(
        xyz,
        animTime,
        waterDepth,
        bumpAmp,
        shoreDir,
        wp
    );

    float waveDisplacement = wave.x;

    // Project the analytic gradient onto the tangent basis to get the
    // height-field slopes along T and B.
    float dhdT = dot(wave.yzw, T);
    float dhdB = dot(wave.yzw, B);

    vec3 bumpedN = normalize(normal - dhdT * T - dhdB * B);
    if (dot(bumpedN, normal) < 0.0) bumpedN = -bumpedN;

    // Displace along the FLAT base normal, not the perturbed one. The analytic
    // normal `bumpedN = N - dHdT*T - dHdB*B` is the exact surface normal only for
    // a height field defined as `base + N * h`. Displacing along the tilted
    // `bumpedN` instead would build a different surface whose true normal no
    // longer matches `bumpedN`, so lighting would disagree with the geometry
    // (visible especially on steep/large waves). Keeping the displacement axis
    // fixed at `normal` makes the rasterized surface and the shading normal
    // consistent.
    pos += waveDisplacement * normal;
    fragNormal = bumpedN;
    fragBaseNormal = normal;   // undisplaced (flat) interpolated base normal
    fragBasePos = vec4(pos - waveDisplacement * normal, bumpAmp);  // base pos + raw amplitude
    fragWaterDepth = waterDepth;

    // Debug: encode displacement as color (normalized against the largest
    // possible envelope: deep + shoal gain + breaker bump).
    float maxExpected = max(bumpAmp * (1.0 + wp.waveShape.w + wp.waveBreaker.x), 1e-3);
    float normDisp = clamp((waveDisplacement / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
    fragDebug = vec3(normDisp);
    
    fragPos = pos;
    fragPosWorld = pos;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(pos, 1.0);
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}
