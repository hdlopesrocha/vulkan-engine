#pragma once

// Material properties per texture (extracted from TextureManager.hpp)
struct MaterialProperties {
    float pomHeightScale = 0.06f;
    float pomMinLayers = 8.0f;
    float pomMaxLayers = 32.0f;
    
    // Mapping mode: 0 = none, 1 = parallax (POM), 2 = tessellation (displacement)
    float mappingMode = 1.0f;
    // When true sample height map directly (white=high). When false use legacy invert (black=deep -> 1.0-height)
    // default: when mappingMode==1 (parallax) we want legacy invert (black=deep)
    bool invertHeight = false;
    // Tessellation-specific height scale (used when mappingMode == 2)
    float tessHeightScale = 0.2f;
    // Tessellation level for hardware displacement (used when mappingMode == 2)
    float tessLevel = 16.0f;
    
    bool flipNormalY = false;
    bool flipTangentHandedness = false;
    float ambientFactor = 0.4f;
    bool flipParallaxDirection = false;
    
    float specularStrength = 0.5f;
    float shininess = 32.0f;
    float padding1 = 0.0f;  // Align to 16 bytes
    float padding2 = 0.0f;

    // Triplanar mapping toggle and per-material UV scales (disabled by default)
    bool triplanar = false;
    float triplanarScaleU = 1.0f;
    float triplanarScaleV = 1.0f;
};
