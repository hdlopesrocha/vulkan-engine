#pragma once

#include "Event.hpp"
#include "../utils/Brush3dManager.hpp"

class SetBrushPaintModeEvent : public Event {
public:
    explicit SetBrushPaintModeEvent(BrushPaintMode mode_) : mode(mode_) {}

    BrushPaintMode mode;
};
