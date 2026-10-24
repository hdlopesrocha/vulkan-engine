#pragma once

// Sub-control selected inside a category. This is what a given input axis /
// button maps to for the active category. UI pages are non-propagating: the
// publisher must not emit application events while on them so that ImGui /
// other UI can consume the raw input without interference.
enum class PageControl {
    TRANSLATE,   // move position (camera or brush)
    ROTATE,      // change orientation (yaw/pitch/roll)
    SCALE,       // change size
    TEXTURE,     // change material / texture
    ATTRIBUTE,   // change SDF-specific attributes
    AIM,         // wiimote M+ orient -> ray octree intersection -> snap brush pos
    COLOR,       // wiimote rotation -> HSV tint on brush
    CONTROL,     // opens outer ring to select Translate/Aim/Scale
    UI,          // non-propagating passthrough (ImGui / UI)
    AZIMUTH,     // light horizontal angle
    ELEVATION    // light vertical angle
};
