#pragma once

#include "Event.hpp"

// Event used to set the selected brush entry's material (texture) index.
class SetBrushTextureEvent : public Event {
public:
    explicit SetBrushTextureEvent(int index_) : index(index_) {}

    int index;
};
