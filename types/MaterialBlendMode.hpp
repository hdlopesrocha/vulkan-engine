#pragma once

#include <cstdint>

// Photoshop-style blend modes applied per material by the video layer composite
// (shaders/includes/BlendModes.glsl). Normal keeps the plain source-over result.
enum class MaterialBlendMode : uint32_t {
    Normal = 0,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    DarkerColor,
    Lighten,
    Screen,
    ColorDodge,
    LighterColor,
    Overlay,
    Difference,
    Exclusion,
    Subtract,
    Divide,
    Hue,
    Saturation,
    Color,
    Luminosity,
};

inline const char* materialBlendModeName(MaterialBlendMode mode) {
    switch (mode) {
        case MaterialBlendMode::Darken: return "Darken";
        case MaterialBlendMode::Multiply: return "Multiply";
        case MaterialBlendMode::ColorBurn: return "Color Burn";
        case MaterialBlendMode::LinearBurn: return "Linear Burn";
        case MaterialBlendMode::DarkerColor: return "Darker Color";
        case MaterialBlendMode::Lighten: return "Lighten";
        case MaterialBlendMode::Screen: return "Screen";
        case MaterialBlendMode::ColorDodge: return "Color Dodge";
        case MaterialBlendMode::LighterColor: return "Lighter Color";
        case MaterialBlendMode::Overlay: return "Overlay";
        case MaterialBlendMode::Difference: return "Difference";
        case MaterialBlendMode::Exclusion: return "Exclusion";
        case MaterialBlendMode::Subtract: return "Subtract";
        case MaterialBlendMode::Divide: return "Divide";
        case MaterialBlendMode::Hue: return "Hue";
        case MaterialBlendMode::Saturation: return "Saturation";
        case MaterialBlendMode::Color: return "Color";
        case MaterialBlendMode::Luminosity: return "Luminosity";
        default: return "Normal";
    }
}
