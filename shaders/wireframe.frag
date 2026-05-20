#version 450

// Minimal wireframe fragment shader: output solid white
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
