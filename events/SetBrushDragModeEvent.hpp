#pragma once

#include "Event.hpp"
#include "../utils/Brush3dManager.hpp"

class SetBrushDragModeEvent : public Event {
public:
    explicit SetBrushDragModeEvent(BrushDragMode mode_) : mode(mode_) {}
    std::string name() const override { return "SetBrushDragModeEvent"; }

    BrushDragMode mode;
};
