#pragma once

#include "Event.hpp"
#include "ControllerContext.hpp"
#include "../utils/Brush3dManager.hpp"
#include <string>

class SetPageEvent : public Event {
public:
    SetPageEvent(PageCategory category_, PageControl control_,
                 BrushControlMode brushMode_ = BrushControlMode::TRANSLATE)
        : category(category_), control(control_), brushMode(brushMode_) {}

    std::string name() const override { return "SetPageEvent"; }

    PageCategory category;
    PageControl control;
    BrushControlMode brushMode;
};
