#include "GamepadPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "ToggleFullscreenEvent.hpp"
#include "CloseWindowEvent.hpp"
#include "RebuildBrushEvent.hpp"

#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "ControllerInput.hpp"

#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/gtc/constants.hpp>

GamepadPublisher::GamepadPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

void GamepadPublisher::update(EventManager* em, const Camera& cam, float deltaTime, ControllerManager* cm, Brush3dManager* brushManager, bool flipRotation) {
    if (!em || !cm) return;

    // Ensure we have a valid gamepad to poll. If the configured joystickId
    // isn't a gamepad, scan for the first available gamepad and use it.
    if (!glfwJoystickIsGamepad(joystickId)) {
        int found = -1;
        for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; ++jid) {
            if (glfwJoystickIsGamepad(jid)) { found = jid; break; }
        }
        if (found == -1) return; // no gamepad connected
        joystickId = found;
    }

    GLFWgamepadstate state;
    if (!glfwGetGamepadState(joystickId, &state)) return;

    ControllerContext& gctx = cm->gamepadContext;
    const ControllerParameters& cp = *cm->getParameters();

    // Button-down toggles: START -> toggle fullscreen, BACK -> close window
    bool startNow = (state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS);
    if (startNow && !startPrev) em->publish(std::make_shared<ToggleFullscreenEvent>());
    startPrev = startNow;

    bool backNow = (state.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS);
    if (backNow && !backPrev) em->publish(std::make_shared<CloseWindowEvent>());
    backPrev = backNow;

    // Use Camera's configured speeds so gamepad feels like keyboard movement.
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    // ---- Page navigation (DPAD) via events ----
    if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::GAMEPAD, PageNavigationEvent::Action::PREV_PAGE));
    if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::GAMEPAD, PageNavigationEvent::Action::NEXT_PAGE));
    if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::GAMEPAD, PageNavigationEvent::Action::PREV_SUBPAGE));
    if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::GAMEPAD, PageNavigationEvent::Action::NEXT_SUBPAGE));

    // Read sticks.
    float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    if (std::abs(lx) < deadzone) lx = 0.0f;
    if (std::abs(ly) < deadzone) ly = 0.0f;
    float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    if (std::abs(rx) < deadzone) rx = 0.0f;
    if (std::abs(ry) < deadzone) ry = 0.0f;

    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    // Triggers -> forward/back translation (right trigger forward, left back).
    float ltrig = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
    float rtrig = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
    float lval = (ltrig + 1.0f) * 0.5f;
    float rval = (rtrig + 1.0f) * 0.5f;
    float net = (rval - lval); // -1..1

    // Bumpers -> roll rotation.
    bool rollL = state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
    bool rollR = state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;

    // ---- Map raw stick/trigger input into an action based on active page ----
    ControllerAction action;
    const PageCategory cat = gctx.activeCategory();
    const PageControl ctrl = gctx.activeControl();

    if (cat == PageCategory::CAMERA) {
        // Camera: translation (stick/triggers) and rotation (right stick/bumpers)
        // are always active, matching the original single-camera-page behaviour.
        if (lx != 0.0f) action.translate += right * (lx * velocity);
        if (ly != 0.0f) action.translate += up * (-ly * velocity); // up stick = up
        if (net != 0.0f) action.translate += forward * (net * velocity);
        action.rotateDeg.x += rotSign * (-rx * angDeg);
        action.rotateDeg.y += rotSign * (-ry * angDeg);
        if (rollL) action.rotateDeg.z += rotSign * (-angDeg);
        if (rollR) action.rotateDeg.z += rotSign * ( angDeg);
    } else { // BRUSH
        float mSpeed = cp.cameraMoveSpeed * deltaTime;
        float aSpeed = cp.cameraAngularSpeedDeg * deltaTime;
        // Transform subpage: combine translate (stick/triggers), rotate
        // (right stick + bumpers) and scale (stick) into one action.
        if (lx != 0.0f) action.translate += right * (lx * mSpeed);
        if (ly != 0.0f) action.translate += up * (-ly * mSpeed);
        if (net != 0.0f) action.translate += forward * (net * mSpeed);
        action.rotateDeg.x += rotSign * (-rx * aSpeed);
        action.rotateDeg.y += rotSign * (-ry * aSpeed);
        if (rollL) action.rotateDeg.z += rotSign * (-aSpeed);
        if (rollR) action.rotateDeg.z += rotSign * ( aSpeed);
        if (lx != 0.0f) action.scaleDelta.x += lx * 0.5f * deltaTime;
        if (ly != 0.0f) action.scaleDelta.y += -ly * 0.5f * deltaTime;
        // Texture / Attribute stay on their dedicated subpages.
        if (ctrl == PageControl::TEXTURE) {
            if (lx > 0.0f) action.textureDelta += 1;
            if (lx < 0.0f) action.textureDelta -= 1;
        } else if (ctrl == PageControl::ATTRIBUTE) {
            if (lx > 0.0f) action.attributeDelta += 1;
            if (lx < 0.0f) action.attributeDelta -= 1;
        }
    }

    bool brushChanged = applyControllerAction(gctx, em, brushManager, action);
    if (brushChanged) em->queue(std::make_shared<RebuildBrushEvent>());
}
