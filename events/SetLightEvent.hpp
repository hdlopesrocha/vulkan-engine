#pragma once

#include "Event.hpp"
#include <string>

class SetLightEvent : public Event {
public:
    SetLightEvent(const std::string& component_, float value_)
        : component(component_), value(value_) {}

    std::string name() const override { return "SetLightEvent"; }

    std::string component; // "Azimuth" or "Elevation"
    float value;
};
