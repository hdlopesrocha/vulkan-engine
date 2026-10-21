#pragma once

#include "Widget.hpp"
#include "types/Settings.hpp"

// Physics controls (GPU water droplets: simulated particles drawn as instanced
// proxy cubes, shaded by an SDF ray-march inside the solid pass). Mirrors
// CloudWidget's structure: a Widget subclass holding the application Settings
// by reference.
class PhysicsWidget : public Widget {
public:
    explicit PhysicsWidget(Settings& settings);

    void render() override;

    // Set by the "Reset now" button; the app reads and clears it (one-shot
    // re-seed request, settings-independent so a preset switch cannot
    // retrigger it).
    bool resetRequested = false;

private:
    Settings& settings;
};
