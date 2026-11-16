#pragma once

#include <cstdint>

inline constexpr uint32_t kWindFieldMaxTornadoes = 4;
inline constexpr uint32_t kWindFieldBinding = 27;
// Octave cap of the shared vector-fBm gust field. Mirrored as
// WIND_FBM_MAX_OCTAVES in shaders/ubo/WindField.glsl (GLSL loop bound) and
// used to clamp the Wind widget / packer value.
inline constexpr uint32_t kWindFbmMaxOctaves = 6;
