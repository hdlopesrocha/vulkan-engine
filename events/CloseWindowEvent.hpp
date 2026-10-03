#pragma once

#include "Event.hpp"

// Event: request the app/window to close
class CloseWindowEvent : public Event {
public:
    CloseWindowEvent() = default;
};