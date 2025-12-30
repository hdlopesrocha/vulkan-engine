// Tangent-Bitangent-Normal (TBN) helper functions

const float TBN_EPSILON = 1e-6;

// Compute tangent and bitangent from vertex tangent attribute (fragTangent.x/y/z, fragTangent.w)
bool computeTBFromVertex(in vec4 fragTangent, in vec3 N, out vec3 T, out vec3 B) {
    if (length(fragTangent.xyz) > TBN_EPSILON) {
        vec3 t = normalize(fragTangent.xyz);
        vec3 b = normalize(cross(N, t) * fragTangent.w);
        T = t;
        B = b;
        return true;
    }
    return false;
}

// Compute tangent and bitangent using screen-space partial derivatives (world-space positions & UVs)
bool computeTBFromDerivatives(in vec3 fragPosWorld, in vec2 fragUV, in vec3 N, out vec3 T, out vec3 B) {
    vec3 dp1 = dFdx(fragPosWorld);
    vec3 dp2 = dFdy(fragPosWorld);
    vec2 duv1 = dFdx(fragUV);
    vec2 duv2 = dFdy(fragUV);
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) > TBN_EPSILON) {
        float r = 1.0 / det;
        vec3 t = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
        vec3 b = normalize((dp2 * duv1.x - dp1 * duv2.x) * r);
        // Orthonormalize against geometry normal
        t = normalize(t - N * dot(N, t));
        b = normalize(cross(N, t));
        T = t;
        B = b;
        return true;
    }
    return false;
}

// Convert sampled tangent-space normal (nmap) into world-space using T/B/N
// NOTE: swap X and Z of the final world normal to match asset convention (fixes X/Z swap observed in rendering)
vec3 normalFromNormalMap(in vec3 nmap, in vec3 T, in vec3 B, in vec3 N) {
    vec3 worldN = normalize(nmap.x * T + nmap.y * B + nmap.z * N);
    // swap X and Z to correct axis mismatch
    return vec3(worldN.z, worldN.y, worldN.x);
}

// Try to compute world-space normal from a normal map using either vertex tangent or derivative-based TBN
bool computeWorldNormalFromNormalMap(in vec4 fragTangent, in vec3 fragPosWorld, in vec2 fragUV, in vec3 N, in vec3 nmap, out vec3 worldNormal, out vec3 T, out vec3 B) {
    // Prefer vertex tangent if available
    if (computeTBFromVertex(fragTangent, N, T, B)) {
        worldNormal = normalFromNormalMap(nmap, T, B, N);
        return true;
    }
    // Fallback to derivatives
    if (computeTBFromDerivatives(fragPosWorld, fragUV, N, T, B)) {
        worldNormal = normalFromNormalMap(nmap, T, B, N);
        return true;
    }
    return false;
}
