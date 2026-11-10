#pragma once

#include "Widget.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../types/BrushEntry.hpp"

// Separate window for the target SDF of a two-SDF brush interpolation. The
// selected brush entry is the source; this window edits its embedded
// BrushSdfTarget (primitive, transform, sweep and SDF parameters).
class BrushInterpolationWidget : public Widget {
public:
    explicit BrushInterpolationWidget(Brush3dManager& manager);

    void render() override;

private:
    Brush3dManager& manager;
};
