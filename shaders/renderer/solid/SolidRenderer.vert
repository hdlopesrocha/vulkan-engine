#version 450
#extension GL_ARB_shader_draw_parameters : require

#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

layout(location = ATTR_POS) in vec3 inPos;
layout(location = ATTR_COLOR) in vec3 inColor;
layout(location = ATTR_UV) in vec2 inUV;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex;
layout(location = ATTR_HSV) in vec3 inHSV;

// ── Solid terrain vertex shader ───────────────────────────────────────────
// Split out of the old shared main.vert (which selected the path with
// -DWATER_MODE). Two compile-time paths:
//   * default (tessellated): writes the TCS interface consumed by SolidRenderer.tesc
//     (includes/solid/SolidTesc.glsl).
//   * SOLID_NO_TESS: writes the fragment interface directly for the
//     TRIANGLE_LIST depth/shadow pipelines; built as shaders/SolidRendererNoTess.vert.spv.

#ifdef SOLID_NO_TESS
// ── Non-tessellation solid VS (shaders/SolidRendererNoTess.vert.spv) ────────────
// Direct VS -> FS interface: writes every varying the DepthOnly.frag and
// EVSM shadow fragment shaders consume, so the TRIANGLE_LIST depth/shadow
// pipelines need no TCS/TES. Valid only when tessellation is off: the TES
// displacement collapses to the undisplaced position then
// (mappingFlag *= tessellationEnabled), and tess levels are 1. Only passes
// whose fragment shaders sample no materials use this VS: the TCS 3-corner
// material compression cannot be reproduced in a VS (one invocation per
// vertex, indexed draws), so the color passes keep the tessellated family to
// preserve slope/height-band blending. The slot/weight outputs below are flat
// provoking-vertex material (exact for single-material triangles) and exist
// only for stage linkage; no bound FS reads them for shading.
// VARY_COLOR output removed (perf report 21 H7): no solid FS reads it, so
// the channel is dropped at linkage like the TCS passthrough. inColor stays
// declared (shared input block) but unused here — the no-tess pipelines below
// must therefore not declare ATTR_COLOR.
layout(location = VARY_UV) out vec2 fragUV;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat out ivec3 fragTexIndices;
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) out vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) out vec3 fragTexWeights;
layout(location = VARY_SHARPNORMAL) out vec3 fragSharpNormal; // unread by every solid FS; written for linkage
layout(location = VARY_HSV) out vec3 fragHSV;
layout(location = VARY_DEBUG) out vec3 fragTessLevel; // 1/16: what the TCS emits when tess is off

void main() {
    fragUV = inUV;
    // Identity model (models removed): inPos is already world space.
    vec3 normal = normalize(inNormal);
    fragNormal = normal;
    vec3 worldPos = inPos;
    fragPosWorld = worldPos;
    fragPosWorldNotDisplaced = worldPos;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(worldPos, 1.0);
    if (inBrushIndex < 0) {
        fragTexIndices = ivec3(0);
        fragTexWeights = vec3(0.0);
    } else {
        int b = inBrushIndex;
        fragTexIndices = ivec3(b, b, b);
        fragTexWeights = vec3(1.0, 0.0, 0.0);
    }
    fragHSV = inHSV;
    fragSharpNormal = normal;
    fragTessLevel = vec3(1.0 / 16.0);
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
}
#else
// ── Tessellated solid VS (shaders/SolidRenderer.vert.spv) ─────────────────────────
layout(location = VARY_COLOR) out vec3 fragColor;
layout(location = VARY_UV) out vec2 fragUV;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) out int fragBrushIndex;      // per-vertex texture index for TCS
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) out vec3 fragLocalPos;          // provide local/world pos to TCS
layout(location = VARY_LOCALNORMAL) out vec3 fragLocalNormal;       // provide local/world normal to TCS
layout(location = VARY_SHARPNORMAL) out vec3 fragSharpNormal;      // face normal
layout(location = VARY_POSCLIP) out vec4 fragPosClip;              // clip-space pos (water back-face pass)
layout(location = VARY_HSV) out vec3 fragHSV;

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Models removed: always use identity model matrix
    mat4 model = mat4(1.0);
    // Transform normal to world space (model is identity here)
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(model) * inNormal);

    // Pass per-vertex texture index as flat int for patch compression in TCS
    fragBrushIndex = inBrushIndex;

    // compute world-space position and pass to fragment
    vec4 worldPos = model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragLocalPos = worldPos.xyz;       // Use world-space position as local basis for displacement

    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;

    // Compute face normal (sharp normal) from model normal
    fragLocalNormal = fragNormal;      // Propagate normal for tessellation stages
    fragSharpNormal = normalize(mat3(model) * inNormal);

    // Pass through per-vertex HSV
    fragHSV = inHSV;

    // apply MVP transform to the vertex position
    gl_Position = ubo.viewProjection * worldPos;
    fragPosClip = gl_Position;
}
#endif
