#pragma once

#include <cstdint>

// CPU mirror of the push-constant block in
// shaders/computation/video/VideoYuvToRgba.comp (see VideoTextureService).
// 96 bytes, inside the 128-byte guaranteed Vulkan push-constant limit.
struct VideoConvertPushConstants {
    int32_t lumaWidth = 0;      // decoded luma width in pixels
    int32_t lumaHeight = 0;     // decoded luma height in pixels
    int32_t chromaWidth = 0;    // lumaWidth / 2  (chroma sample count)
    int32_t chromaHeight = 0;   // lumaHeight / 2 (chroma sample count)
    int32_t dstWidth = 0;       // texture array layer width
    int32_t dstHeight = 0;      // texture array layer height
    int32_t padding0 = 0;
    int32_t padding1 = 0;
    // rgb = matrix * (Y, U, V) + offset (offset in xyz; w unused)
    float offset[3] = {0.0f, 0.0f, 0.0f};
    float padding2 = 0.0f;
    float matrixRow0[3] = {1.0f, 0.0f, 0.0f};
    float padding3 = 0.0f;
    float matrixRow1[3] = {0.0f, 1.0f, 0.0f};
    float padding4 = 0.0f;
    float matrixRow2[3] = {0.0f, 0.0f, 1.0f};
    float padding5 = 0.0f;
};
static_assert(sizeof(VideoConvertPushConstants) == 96,
              "VideoConvertPushConstants must match the shader push-constant block");
