#pragma once

#include <cstdint>

// Shared shader location constants (must match shaders/includes/locations.glsl)
// Keep these in sync with the GLSL header to avoid mismatches.
static constexpr uint32_t ATTR_POS = 0u;
static constexpr uint32_t ATTR_COLOR = 1u;
static constexpr uint32_t ATTR_UV = 2u;
static constexpr uint32_t ATTR_NORMAL = 3u;
static constexpr uint32_t ATTR_BRUSH_INDEX = 4u;
static constexpr uint32_t ATTR_INSTANCE = 5u;

// Fragment output locations (match render pass attachments)
static constexpr uint32_t FRAG_OUT_COLOR = 0u;
static constexpr uint32_t FRAG_OUT_NORMAL = 1u;

// Inter-stage varyings (compact contiguous range to match shaders/includes/locations.glsl)
static constexpr uint32_t VARY_COLOR = 22u;
static constexpr uint32_t VARY_UV = 7u;
static constexpr uint32_t VARY_NORMAL = 8u;
static constexpr uint32_t VARY_POSWORLD = 9u;
static constexpr uint32_t VARY_BRUSHPATCH = 10u;
static constexpr uint32_t VARY_POSLIGHT = 11u;
// geometry-used varyings kept low to avoid maxGeometryInputComponents overflow
static constexpr uint32_t VARY_PLANE_NORMAL = 12u;
static constexpr uint32_t VARY_FACE_NORMAL = 13u;
static constexpr uint32_t VARY_ROTFRAC = 14u;
static constexpr uint32_t VARY_LOCALNORMAL = 15u;
static constexpr uint32_t VARY_TEXWEIGHTS = 16u;
static constexpr uint32_t VARY_POSCLIP = 17u;
static constexpr uint32_t VARY_DEBUG = 18u;
static constexpr uint32_t VARY_TANGENTWS = 19u;
static constexpr uint32_t VARY_LOCALPOS = 20u;
static constexpr uint32_t VARY_SDF = 21u;
static constexpr uint32_t VARY_SHARPNORMAL = 6u;
