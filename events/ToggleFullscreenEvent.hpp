#pragma once

#include "Event.hpp"

// Event: toggle fullscreen state (no payload)
class ToggleFullscreenEvent : public Event {
public:
    ToggleFullscreenEvent() = default;
};
