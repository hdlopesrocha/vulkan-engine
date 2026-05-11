#version 450

layout(location = 0) in float fragSdf;

layout(location = 0) out vec4 outColor;

void main() {
    if (isnan(fragSdf) || isinf(fragSdf) || abs(fragSdf) > 10.0) {
        discard;
    }

    float normalizedMagnitude = clamp(abs(fragSdf) / 10.0, 0.0, 1.0);
    vec3 negativeColor = vec3(1.0, 0.0, 0.0);
    vec3 positiveColor = vec3(0.0, 1.0, 0.0);
    vec3 color = fragSdf >= 0.0 ? positiveColor : negativeColor;

    float surfaceWidth = max(fwidth(fragSdf) * 1.5, 0.015);
    float surfaceLine = 1.0 - smoothstep(0.0, surfaceWidth, abs(fragSdf));

    color *= 0.35 + 0.65 * (1.0 - normalizedMagnitude);
    color = mix(color, vec3(1.0), surfaceLine);
    outColor = vec4(color, 1.0);
}
