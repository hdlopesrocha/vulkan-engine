#pragma once

// Material properties per texture (extracted from TextureManager.hpp)
struct MaterialProperties {
    // Mapping enabled: false = disabled, true = enable tessellation + bump mapping
    bool mappingMode = false;
    // When true sample height map directly (white=high). When false use legacy invert (black=deep -> 1.0-height)
    bool invertHeight = false;
    // Tessellation-specific height scale (used when mappingMode == 2)
    float tessHeightScale = 0.2f;
    // Tessellation level for hardware displacement (used when mappingMode == 2)
    float tessLevel = 16.0f;
    
    float ambientFactor = 0.4f;
    
    float specularStrength = 0.5f;
    float shininess = 32.0f;
    float padding1 = 0.0f;  // Align to 16 bytes
    float padding2 = 0.0f;

    // Triplanar mapping toggle and per-material UV scales (disabled by default)
    bool triplanar = false;
    float triplanarScaleU = 1.0f;
    float triplanarScaleV = 1.0f;

    // Normal map conventions (defaults keep existing behavior)
    bool normalFlipY = false;   // flip normal map Y (green) channel if true
    bool normalSwapXZ = false;  // swap R/B channels (X/Z) if true
        // default: legacy invert behavior for backward compatibility
};
