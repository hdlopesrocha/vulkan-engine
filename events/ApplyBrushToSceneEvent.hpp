#pragma once

#include "Event.hpp"

// Event used to request the current brush be applied to the main scene's
// octree on the corresponding layer (opaque or transparent as selected
// in the brush entry). Published by keyboard (Space) or wiimote (B).
class ApplyBrushToSceneEvent : public Event {
public:
    ApplyBrushToSceneEvent() = default;
    std::string name() const override { return "ApplyBrushToSceneEvent"; }
};
