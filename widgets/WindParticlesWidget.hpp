#pragma once

#include "Widget.hpp"

class Settings;
class VegetationRenderer;

// Wind-trail particle editor (WindRenderer).
//
// Presentation controls (count, trail duration, color, width, spawn band) live
// in Settings::windParticles; the wind parameters edited below are the SHARED
// field owned by VegetationRenderer (the same sliders as the Wind widget), so
// these particles, the grass and the clouds always follow one wind.
class WindParticlesWidget : public Widget {
public:
    WindParticlesWidget(Settings* settings_, VegetationRenderer* vegetationRenderer_);
    void render() override;

private:
    Settings* settings;
    VegetationRenderer* vegetationRenderer;
};
