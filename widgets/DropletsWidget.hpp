#pragma once

#include "Widget.hpp"
#include "types/Settings.hpp"

// Droplet renderer controls (GPU water droplets: simulated particles drawn as
// instanced proxy cubes, shaded by an SDF ray-march inside the solid pass).
// Mirrors CloudWidget's structure: a Widget subclass holding the application
// Settings by reference.
class DropletsWidget : public Widget {
public:
    explicit DropletsWidget(Settings& settings);

    void render() override;

    // Set by the "Reset now" button; the app reads and clears it (one-shot
    // re-seed request, settings-independent so a preset switch cannot
    // retrigger it).
    bool resetRequested = false;

private:
    Settings& settings;
};
