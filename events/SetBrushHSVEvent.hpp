#pragma once

#include "Event.hpp"
#include <string>

class SetBrushHSVEvent : public Event {
public:
    SetBrushHSVEvent(const std::string& component_, float value_)
        : component(component_), value(value_) {}

    std::string name() const override { return "SetBrushHSVEvent"; }

    std::string component; // "Hue", "Saturation", or "Value"
    float value;
};
