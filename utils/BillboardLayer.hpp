#pragma once


// Represents a single layer in a billboard
struct BillboardLayer {
    int atlasIndex = 0;        // Which atlas texture (0=foliage, 1=grass, 2=wild)
    int tileIndex = 0;         // Which tile from that atlas
    
    // Transform properties
    float offsetX = 0.0f;      // Position offset in billboard space (-1 to 1)
    float offsetY = 0.0f;
    float scaleX = 1.0f;       // Scale (1.0 = full size)
    float scaleY = 1.0f;
    float rotation = 0.0f;     // Rotation in degrees (0-360)
    
    // Rendering properties
    float opacity = 1.0f;      // Alpha multiplier (0-1)
    int renderOrder = 0;       // Higher values render on top
};
