#pragma once

#include "Event.hpp"
#include "types/GraphicsQuality.hpp"

// Queued by the main-UI quality buttons; MyApp runs a GraphicsSettingsCommand
// for the requested preset when the event is drained.
class SetGraphicsQualityEvent : public Event {
public:
    explicit SetGraphicsQualityEvent(GraphicsQuality quality_) : quality(quality_) {}

    GraphicsQuality quality;
};
