#pragma once

#include "Event.hpp"

// Event used to request a deferred brush-space rebuild from the main application
class RebuildBrushEvent : public Event {
public:
    RebuildBrushEvent() = default;
    std::string name() const override { return "RebuildBrushEvent"; }
};
