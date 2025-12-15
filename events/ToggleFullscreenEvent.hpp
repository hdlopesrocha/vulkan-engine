#pragma once

#include "Event.hpp"
#include <string>

// Event: toggle fullscreen state (no payload)
class ToggleFullscreenEvent : public Event {
public:
    ToggleFullscreenEvent() = default;
    std::string name() const override { return "ToggleFullscreenEvent"; }
};
