#ifndef R_T_PAYLOAD_GLSL
#define R_T_PAYLOAD_GLSL

// Extracted from shaders/includes/rt_params.glsl (single-struct GLSL type).

// Shared ray payload (rgen + rmiss + rchit). MUST stay a single variable:
// SPIR-V allows at most one IncomingRayPayloadKHR per entry point
// (VUID-StandaloneSpirv-IncomingRayPayloadKHR-04700).
struct RTPayload {
    vec3 color;        // hit color (or sky on miss)
    float hitDistance; // hitT (or -1 on miss)
    float coarseF;     // 0=fine box (use hit as-is), 1=coarse (feather to deep/sky)
};

#endif // R_T_PAYLOAD_GLSL
