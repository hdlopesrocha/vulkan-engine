// Shadow calculation with PCF (Percentage Closer Filtering)
float ShadowCalculation(vec4 fragPosLightSpace, float bias) {
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // With GLM_FORCE_DEPTH_ZERO_TO_ONE, Z is already in [0,1]
    // But XY still need transformation from [-1,1] to [0,1]
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    
    // Outside shadow map bounds = no shadow
    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;
    
    // Get closest depth value from light's perspective
    float currentDepth = projCoords.z;
    
    // PCF (Percentage Closer Filtering) for soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    
    // Use larger 5x5 kernel to blur over gaps at cube edges
    for(int x = -2; x <= 2; ++x)
    {
        for(int y = -2; y <= 2; ++y)
        {
            vec2 offset = vec2(x, y) * texelSize;
            float pcfDepth = texture(shadowMap, projCoords.xy + offset).r;
            // Shadow test with bias to prevent shadow acne
            shadow += currentDepth > (pcfDepth + bias) ? 1.0 : 0.0;
        }
    }
    shadow /= 25.0; // Average the 25 samples (5x5)
    
    return shadow;
}