#pragma once

#include "Event.hpp"
#include <string>

// Identifies which physical controller a context / navigation event belongs to.
enum class ControllerId { KEYBOARD, MOUSE, GAMEPAD, WIIMOTE };

// Event used to switch pages. Publishing it lets ANY controller (e.g. the
// keyboard) switch the pages of ANY other controller (e.g. the mouse). Target
// contexts react in their IEventHandler::onEvent.
class PageNavigationEvent : public Event {
public:
    enum class Action { NEXT_PAGE, PREV_PAGE, NEXT_SUBPAGE, PREV_SUBPAGE };

    PageNavigationEvent(ControllerId target_, Action action_)
        : target(target_), action(action_) {}

    std::string name() const override { return "PageNavigationEvent"; }

    ControllerId target;
    Action action;
};
