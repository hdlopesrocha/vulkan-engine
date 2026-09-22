#pragma once

#include "Event.hpp"
#include "../utils/GraphicsQuality.hpp"

// Queued by the main-UI quality buttons; MyApp runs a GraphicsSettingsCommand
// for the requested preset when the event is drained.
class SetGraphicsQualityEvent : public Event {
public:
    explicit SetGraphicsQualityEvent(GraphicsQuality quality_) : quality(quality_) {}

    std::string name() const override { return "SetGraphicsQualityEvent"; }

    GraphicsQuality quality;
};
