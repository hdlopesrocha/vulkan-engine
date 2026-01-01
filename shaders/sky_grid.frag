#version 450

layout(location = 0) in vec3 fragPosWorld;
layout(location = 1) in vec3 fragNormal;

#include "includes/ubo.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    // Direction from camera to fragment (world-space)
    vec3 viewDir = normalize(fragPosWorld - ubo.viewPos.xyz);
    
    // Grid parameters
    float gridScale = 5.0; // spacing between grid lines (in degrees or similar)
    float lineWidth = 0.015; // line thickness
    float axisWidth = 0.03; // axis line thickness (thicker than grid)
    
    // Convert view direction to spherical-like coordinates for grid
    // Use viewDir components directly for grid mapping
    vec3 absDir = abs(viewDir);
    
    // Create grid lines based on viewDir components
    vec3 gridPos = viewDir * gridScale;
    vec3 gridFract = fract(gridPos + 0.5) - 0.5; // centered on grid lines
    vec3 gridDist = abs(gridFract);
    
    // Determine which components are close to grid lines
    float gridX = smoothstep(lineWidth, lineWidth * 0.5, gridDist.x);
    float gridY = smoothstep(lineWidth, lineWidth * 0.5, gridDist.y);
    float gridZ = smoothstep(lineWidth, lineWidth * 0.5, gridDist.z);
    
    // Combine grid lines (show on multiple planes)
    float grid = 0.0;
    // YZ plane grid (shows when looking along X)
    grid = max(grid, gridY * (1.0 - absDir.x * 0.7));
    grid = max(grid, gridZ * (1.0 - absDir.x * 0.7));
    // XZ plane grid (shows when looking along Y)
    grid = max(grid, gridX * (1.0 - absDir.y * 0.7));
    grid = max(grid, gridZ * (1.0 - absDir.y * 0.7));
    // XY plane grid (shows when looking along Z)
    grid = max(grid, gridX * (1.0 - absDir.z * 0.7));
    grid = max(grid, gridY * (1.0 - absDir.z * 0.7));
    
    // Axes colors using normal visualization colors
    // Map viewDir from [-1,1] to [0,1] for RGB (standard normal color mapping)
    // X component -> Red, Y component -> Green, Z component -> Blue
    
    vec3 axisColor = vec3(0.0);
    float axisMask = 0.0;
    
    // Detect proximity to each axis
    float nearXAxis = smoothstep(axisWidth, axisWidth * 0.5, length(viewDir.yz));
    float nearYAxis = smoothstep(axisWidth, axisWidth * 0.5, length(viewDir.xz));
    float nearZAxis = smoothstep(axisWidth, axisWidth * 0.5, length(viewDir.xy));
    
    // Convert view direction to normal colors (map from [-1,1] to [0,1])
    vec3 normalColor = viewDir * 0.5 + 0.5;
    
    // X axis - use full normal color along X direction
    if (nearXAxis > 0.01) {
        axisColor = mix(axisColor, normalColor, nearXAxis);
        axisMask = max(axisMask, nearXAxis);
    }
    
    // Y axis - use full normal color along Y direction
    if (nearYAxis > 0.01) {
        axisColor = mix(axisColor, normalColor, nearYAxis);
        axisMask = max(axisMask, nearYAxis);
    }
    
    // Z axis - use full normal color along Z direction
    if (nearZAxis > 0.01) {
        axisColor = mix(axisColor, normalColor, nearZAxis);
        axisMask = max(axisMask, nearZAxis);
    }
    
    // Base background color (dark gray)
    vec3 backgroundColor = vec3(0.05, 0.05, 0.08);
    
    // Grid color (light gray)
    vec3 gridColor = vec3(0.3, 0.3, 0.35);
    
    // Combine: background -> grid -> axes
    vec3 color = backgroundColor;
    color = mix(color, gridColor, grid * 0.6);
    color = mix(color, axisColor, axisMask);
    
    // Add subtle gradient to make it more visually interesting
    float gradient = smoothstep(-0.5, 0.5, viewDir.y) * 0.1;
    color += vec3(gradient);
    
    outColor = vec4(color, 1.0);
}
