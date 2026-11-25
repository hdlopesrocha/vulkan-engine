#pragma once

// Base class for events. Users can derive their own event types from this.
class Event {
public:
    Event() = default;
    virtual ~Event() = default;
};
