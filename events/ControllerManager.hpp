#pragma once

#include "ControllerParameters.hpp"
#include "ControllerContext.hpp"
#include "EventManager.hpp"

class ControllerManager {
public:
    ControllerManager() {
        keyboardContext.buildDefaultTree();
        mouseContext.buildDefaultTree();
        gamepadContext.buildDefaultTree();
        wiimoteContext.buildDefaultTree();

        // Mouse defaults to the Camera > Transform page so left-drag rotates
        // the camera out of the box. Mouse-camera events are suppressed while
        // ImGui is capturing the mouse (see MousePublisher), so ImGui windows
        // keep working perfectly. The non-propagating UI subpage is still
        // reachable via keys 7/8 for total passthrough.
        mouseContext.selectControl(PageControl::TRANSLATE);
    }

    // Owned speed/tunable parameters (shared across controllers).
    ControllerParameters parameters;

    // One independent page tree per controller.
    ControllerContext keyboardContext{ControllerId::KEYBOARD};
    ControllerContext mouseContext{ControllerId::MOUSE};
    ControllerContext gamepadContext{ControllerId::GAMEPAD};
    ControllerContext wiimoteContext{ControllerId::WIIMOTE};

    ControllerParameters* getParameters() { return &parameters; }
    const ControllerParameters* getParameters() const { return &parameters; }

    ControllerContext& context(ControllerId id) {
        switch (id) {
            case ControllerId::KEYBOARD: return keyboardContext;
            case ControllerId::MOUSE:    return mouseContext;
            case ControllerId::GAMEPAD:  return gamepadContext;
            case ControllerId::WIIMOTE:  return wiimoteContext;
        }
        return keyboardContext;
    }
    const ControllerContext& context(ControllerId id) const {
        switch (id) {
            case ControllerId::KEYBOARD: return keyboardContext;
            case ControllerId::MOUSE:    return mouseContext;
            case ControllerId::GAMEPAD:  return gamepadContext;
            case ControllerId::WIIMOTE:  return wiimoteContext;
        }
        return keyboardContext;
    }

    // Subscribe every per-controller context so PageNavigationEvents can reach them.
    void subscribeContexts(EventManager& em) {
        em.subscribe(&keyboardContext);
        em.subscribe(&mouseContext);
        em.subscribe(&gamepadContext);
        em.subscribe(&wiimoteContext);
    }

    // Switch all four controller contexts to the given page category and control.
    void switchAllContexts(PageCategory cat, PageControl ctrl) {
        keyboardContext.selectPage(cat, ctrl);
        mouseContext.selectPage(cat, ctrl);
        gamepadContext.selectPage(cat, ctrl);
        wiimoteContext.selectPage(cat, ctrl);
    }

    // Switch all four controller contexts to Brush page with the given subpage control.
    void switchAllToBrush(PageControl ctrl) {
        switchAllContexts(PageCategory::BRUSH, ctrl);
    }
};
