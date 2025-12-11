#pragma once

#include <string>
#include <memory>

// Base class for events. Users can derive their own event types from this.
class Event {
public:
    using Ptr = std::shared_ptr<Event>;
    Event() = default;
    virtual ~Event() = default;

    // A short runtime name useful for logging/debugging
    virtual std::string name() const { return "Event"; }
};

// Helper to create events without having to spell std::make_shared everywhere
template<typename T, typename... Args>
inline std::shared_ptr<T> make_event(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}
