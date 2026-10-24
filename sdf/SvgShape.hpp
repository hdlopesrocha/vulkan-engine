#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "SvgContour.hpp"

// Minimal dependency-free SVG loader for the SVG brush
// (sdf/SvgPolygonDistanceFunction + utils/SvgBrush).
//
// Supported elements: <path> (M/m L/l H/h V/v C/c S/s Q/q T/t A/a Z/z),
// <rect>, <circle>, <ellipse>, <polygon>, <polyline>, <line>. Group elements
// (<g>) contribute their transform to nested elements; per-element
// transform="translate/scale/rotate/matrix" and style="fill:..." are applied.
// Fills: #rgb, #rrggbb, rgb()/rgba(), a small named-colour table; `none` and
// unparsable colours fall back to white (the brush still tints per contour).
// Curves and arcs are flattened into polylines.
//
// Coordinates are normalized into [-1,1]^2 using the viewBox (or width/height):
// SVG +x maps to local +x and SVG +y maps to local -z, so a drawing appears
// upright when looked at from +Y in the model's local frame.
class SvgShape {
public:
    std::vector<SvgContour> contours;
    glm::vec2 viewMin = glm::vec2(0.0f);
    glm::vec2 viewSize = glm::vec2(1.0f);
    std::string source;  // path or "<string>" for diagnostics

    // Load + parse. Returns false and fills `error` on I/O or parse failure
    // (a parse failure with zero contours is an error; a valid but empty SVG
    // document yields true with no contours).
    static bool loadFromFile(const std::string& path, SvgShape& out, std::string& error);
    // Same, but also tries the usual runtime locations for a relative path
    // (the app runs from bin/): `path`, `bin/<path>` and `../<path>`.
    static bool loadFromFileResolved(const std::string& path, SvgShape& out, std::string& error,
                                     std::string* resolvedPath = nullptr);
    static bool fromString(const std::string& text, SvgShape& out, std::string& error);

    // Index of the contour that contains p (the last matched contour wins so
    // later/drawn-on-top shapes colour the pixel), or -1.
    int containingContour(const glm::vec2& p) const;

    static bool parseColor(const std::string& text, glm::vec3& rgb);
};
