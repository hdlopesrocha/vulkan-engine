#ifndef BRUSH_MAP_GLSL
#define BRUSH_MAP_GLSL

// Per-brush image source table (types/ImageType.hpp), uploaded as one uvec2 per
// brush: x = ImageType (0 = Image, 1 = TextureMixer, 2 = Video), y = the index
// inside that source (a texture-array layer for Image/TextureMixer, a video slot
// for Video). It decouples the per-vertex brush index (material index) from the
// texture layer it samples, so a brush can point at a base image, a
// TextureMixer output, or a video clip.
//
// Included via SceneBindings.glsl so every material-sampling shader sees the
// binding and the resolve helper.
layout(std430, set = 0, binding = 28) readonly buffer BrushMapBuffer {
    uvec2 brushMap[];
};

const uint IMAGE_TYPE_IMAGE = 0u;
const uint IMAGE_TYPE_TEXTURE_MIXER = 1u;
const uint IMAGE_TYPE_VIDEO = 2u;

// The sampled texture-array layer for a brush. Image and TextureMixer both live
// in the 5-map material array; Video is sampled from the dedicated video array
// (handled by the video layer / VideoRenderer.frag, which reads brushVideoSlot()).
// A Video brush that reaches the solid/water path falls back to base layer 0 so
// the lookup can never go out of range.
int brushLayer(in int brush) {
    uvec2 image = brushMap[uint(max(brush, 0))];
    return (image.x == IMAGE_TYPE_VIDEO) ? 0 : int(image.y);
}

// True when the brush samples a decoded video clip.
bool brushIsVideo(in int brush) {
    return brushMap[uint(max(brush, 0))].x == IMAGE_TYPE_VIDEO;
}

// The video slot a Video brush samples (0 for non-video brushes).
int brushVideoSlot(in int brush) {
    return int(brushMap[uint(max(brush, 0))].y);
}

#endif // BRUSH_MAP_GLSL
