#pragma once

// How a GPU-only secondary SDF combines with the primary octree SDF.
// `Displace` is the historical behavior: the pattern is converted to a signed
// surface offset and folded into the field as F = d_primary - h.
// The CSG modes treat the pattern as a signed distance field (negative inside)
// and combine it directly: union = min, intersection = max,
// subtraction = max(primary, -secondary).
enum class SecondarySdfOperation : int {
    Displace = 0,
    Union = 1,
    Intersection = 2,
    Subtraction = 3
};
