#pragma once

#include "Event.hpp"
#include <string>

// Event: request the app/window to close
class CloseWindowEvent : public Event {
public:
    CloseWindowEvent() = default;
    std::string name() const override { return "CloseWindowEvent"; }
};

// Event: toggle fullscreen state (no payload)
class ToggleFullscreenEvent : public Event {
public:
    ToggleFullscreenEvent() = default;
    std::string name() const override { return "ToggleFullscreenEvent"; }
};
