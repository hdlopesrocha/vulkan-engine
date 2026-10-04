#pragma once

// Graphics-quality presets exposed by the main UI ("Maximum" / "Minimal").
// Shared by the UI event and the command that applies the preset, so neither
// layer has to depend on the other.
enum class GraphicsQuality {
    Maximum,
    Minimal
};
