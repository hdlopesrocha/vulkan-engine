#pragma once

#include "Event.hpp"

// Event used to set the selected brush entry's SDF type index.
class SetBrushSdfTypeEvent : public Event {
public:
    explicit SetBrushSdfTypeEvent(int sdfType_) : sdfType(sdfType_) {}

    int sdfType;
};
