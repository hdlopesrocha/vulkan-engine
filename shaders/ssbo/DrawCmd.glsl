#ifndef DRAW_CMD_GLSL
#define DRAW_CMD_GLSL

// Extracted from shaders/IndirectRenderer.comp (single-struct GLSL type).

struct DrawCmd {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

#endif // DRAW_CMD_GLSL
