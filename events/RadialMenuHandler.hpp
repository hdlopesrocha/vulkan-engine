#pragma once

#include <memory>
#include <string>
#include <vector>
#include <GLFW/glfw3.h>
#include "ControllerContext.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../widgets/RadialMenu.hpp"

class EventManager;
class NunchukPublisher;
class GamepadPublisher;
class ControllerManager;
class Brush3dManager;
class TextureArrayManager;
class Light;

// Controller-agnostic radial menu input handler.
// Encapsulates toggle detection, input routing, and ring interaction
// for all input sources (keyboard, gamepad, nunchuk, mouse).
class RadialMenuHandler {
public:
    enum class LabelRingKind { CONTROL, PAINT, DRAG, HSV, LIGHT, SHAPE };

    RadialMenuHandler(
        GLFWwindow* window,
        EventManager* eventManager,
        RadialMenu* radialMenu,
        NunchukPublisher* nunchuk,
        GamepadPublisher* gamepad,
        ControllerManager* controllerManager,
        Brush3dManager* brushManager,
        TextureArrayManager* textureArrayManager,
        Light* light
    );

    // Process radial menu toggle and input for the current frame.
    // Returns true if the radial menu is visible (callers should suppress
    // normal input when this returns true).
    bool update(uint32_t loadedTextureLayers);

    // Build and set the page tree on the radial menu.
    void setupPages();

    bool isHomePrev() const { return homePrev; }
    bool isMiddleMousePrev() const { return middleMousePrev; }

private:
    // Detect toggle inputs (Home key, middle mouse, nunchuk Home, gamepad Start).
    void detectToggle();
    // Feed the correct input vector based on which controller is connected.
    void feedInputVector();
    // Detect select/back edge inputs from the primary controller.
    void detectSelectBack();
    // Route interactions on the active ring.
    void routeActiveRing();
    // Route subpage selection to the appropriate ring push.
    void routeSubpageSelection(int stackPage, int hoveredSub);
    // Queue a SetPageEvent for all controllers.
    void queueSetPageEvent(PageCategory cat, PageControl ctrl, BrushControlMode bm);

    GLFWwindow* window_;
    EventManager* em_;
    RadialMenu* menu_;
    NunchukPublisher* nunchuk_;
    GamepadPublisher* gamepad_;
    ControllerManager* cm_;
    Brush3dManager* brush_;
    TextureArrayManager* texArray_;
    Light* light_;

    // Edge-detection state
    bool homePrev = false;
    bool middleMousePrev = false;
    bool textureSelectPrev = false;
    bool backPrev = false;

    // Label ring kind (which label ring is currently displayed)
    LabelRingKind labelRingKind = LabelRingKind::CONTROL;

    // Page tree (owned by handler, set on RadialMenu)
    std::vector<Page> pages_;

    std::vector<ImTextureID> textureScratch_;
};
