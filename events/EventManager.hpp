#pragma once

#include "Event.hpp"
#include "IEventHandler.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <functional>

// Simple thread-safe event manager. Handlers can subscribe/unsubscribe and events can be
// published immediately or queued for later processing.
class EventManager {
public:
    using EventPtr = std::shared_ptr<Event>;
    using HandlerPtr = IEventHandler*; // non-owning pointer (caller manages lifetime)

    EventManager() = default;
    ~EventManager();

    // Register a handler to receive events. Duplicate registrations are ignored.
    void subscribe(HandlerPtr handler);

    // Unregister a handler.
    void unsubscribe(HandlerPtr handler);

    // Publish an event immediately to all subscribed handlers.
    void publish(const EventPtr &event);

    // Queue an event for later processing. The event will be processed when processQueued() is called.
    void queue(const EventPtr &event);

    // Process all queued events (in FIFO order) and dispatch them to handlers.
    void processQueued();

private:
    std::mutex handlersMutex;
    std::vector<HandlerPtr> handlers; // guarded by handlersMutex

    std::mutex queueMutex;
    std::queue<EventPtr> eventQueue; // guarded by queueMutex
};
