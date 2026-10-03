#pragma once

#include <memory>

// Base class for events. Users can derive their own event types from this.
class Event {
public:
    using Ptr = std::shared_ptr<Event>;
    Event() = default;
    virtual ~Event() = default;
};
