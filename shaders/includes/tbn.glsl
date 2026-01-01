// Tangent-Bitangent-Normal (TBN) helper functions

const float TBN_EPSILON = 1e-6;

// Apply per-material normal map conventions: flip Y and swap X/Z channels
vec3 applyNormalConvention(in vec3 n, in vec4 normalParams) {
    vec3 nn = n;
    if (normalParams.x > 0.5) nn.y = -nn.y; // flip Y
    if (normalParams.y > 0.5) nn = vec3(nn.z, nn.y, nn.x); // swap X/Z
    return nn;
}



