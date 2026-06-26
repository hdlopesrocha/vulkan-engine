#pragma once

// Material properties per texture (extracted from TextureManager.hpp)
struct MaterialProperties {
    // Mapping enabled: false = disabled, true = enable tessellation + bump mapping
    bool mappingMode = false;
    // Mirror height map V coordinate vertically when true
    bool invertHeight = false;
    // Tessellation-specific height scale (used when mappingMode == 2)
    float tessHeightScale = 0.2f;
    // Tessellation level for hardware displacement (used when mappingMode == 2)
    float tessLevel = 16.0f;
    // Per-material distance-based tessellation range
    float tessMinLevel = 4.0f;
    float tessMaxLevel = 32.0f;
    
    float ambientFactor = 0.4f;
    
    float specularStrength = 0.5f;
    float shininess = 32.0f;
    float reflectionStrength = 0.0f;

    // Triplanar mapping toggle and per-material UV scales (disabled by default)
    bool triplanar = true;
    float triplanarScaleU = 0.01f;
    float triplanarScaleV = 0.01f;

    // Flip height map U coordinate horizontally
    bool invertWidth = false;
    // Normal map conventions (defaults keep existing behavior)
    bool normalFlipY = false;   // flip normal map Y (green) channel if true
    bool normalSwapXZ = false;  // swap R/B channels (X/Z) if true
        // default: legacy invert behavior for backward compatibility

    float roughnessFactor = 0.5f;
    float aoFactor = 1.0f;
    bool useAO = true;
};
