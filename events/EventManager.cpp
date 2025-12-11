#include "EventManager.hpp"
#include <algorithm>
#include <iostream>

EventManager::~EventManager() {
    // Clear queued events
    std::lock_guard<std::mutex> ql(queueMutex);
    while (!eventQueue.empty()) eventQueue.pop();
    // Note: we don't own handlers; just clear the list
    std::lock_guard<std::mutex> hl(handlersMutex);
    handlers.clear();
}

void EventManager::subscribe(HandlerPtr handler) {
    if (!handler) return;
    std::lock_guard<std::mutex> lock(handlersMutex);
    auto it = std::find(handlers.begin(), handlers.end(), handler);
    if (it == handlers.end()) handlers.push_back(handler);
}

void EventManager::unsubscribe(HandlerPtr handler) {
    if (!handler) return;
    std::lock_guard<std::mutex> lock(handlersMutex);
    handlers.erase(std::remove(handlers.begin(), handlers.end(), handler), handlers.end());
}

void EventManager::publish(const EventPtr &event) {
    if (!event) return;
    // make a snapshot of handlers to avoid holding the lock while invoking callbacks
    std::vector<HandlerPtr> snapshot;
    {
        std::lock_guard<std::mutex> lock(handlersMutex);
        snapshot = handlers;
    }
    for (auto h : snapshot) {
        try {
            if (h) h->onEvent(event);
        } catch (const std::exception &e) {
            std::cerr << "Event handler threw exception: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "Event handler threw unknown exception\n";
        }
    }
}

void EventManager::queue(const EventPtr &event) {
    if (!event) return;
    std::lock_guard<std::mutex> lock(queueMutex);
    eventQueue.push(event);
}

void EventManager::processQueued() {
    // Drain queue into a local list to minimize lock time
    std::vector<EventPtr> events;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        while (!eventQueue.empty()) {
            events.push_back(eventQueue.front());
            eventQueue.pop();
        }
    }
    if (events.empty()) return;

    // Dispatch each event
    for (auto &e : events) publish(e);
}
