#pragma once

#include "Event.hpp"
#include "../utils/Brush3dManager.hpp"

// Event to set the active brush control mode across all controller contexts.
class SetBrushControlEvent : public Event {
public:
    explicit SetBrushControlEvent(BrushControlMode mode_) : mode(mode_) {}

    BrushControlMode mode;
};
