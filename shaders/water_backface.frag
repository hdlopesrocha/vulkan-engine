#version 450

// Depth-only pass for water back-face rendering.
// Uses the same vertex + tessellation shaders as the main water pass.
// Only depth is written by the rasterizer; no color output needed.

void main() {
    // Intentionally empty — depth is recorded automatically.
}
