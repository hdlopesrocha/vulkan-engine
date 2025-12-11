#pragma once

#include "Event.hpp"
#include <memory>

// Interface for event handlers. Implement this and register with EventManager to receive events.
class IEventHandler {
public:
    using EventPtr = std::shared_ptr<Event>;
    virtual ~IEventHandler() = default;

    // Called when an event is published. Implementations should be quick; heavy work can be deferred.
    virtual void onEvent(const EventPtr &event) = 0;
};
