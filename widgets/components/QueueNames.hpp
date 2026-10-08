#pragma once

// Canonical display names for the logical GPU queues. Both queue widgets
// reference these constants instead of repeating string literals, so the role
// names cannot drift apart.
namespace QueueNames {

inline constexpr const char* Graphics     = "Graphics";
inline constexpr const char* Present      = "Present";
inline constexpr const char* Solid        = "Solid";
inline constexpr const char* Water        = "Water";
inline constexpr const char* Vegetation   = "Vegetation";
inline constexpr const char* SDF          = "SDF";
inline constexpr const char* BoundingBox  = "BoundingBox";
inline constexpr const char* Geometry     = "Geometry";
inline constexpr const char* Transfer     = "Transfer";
inline constexpr const char* Brush        = "Brush";

} // namespace QueueNames
