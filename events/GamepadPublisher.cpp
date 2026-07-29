#include "GamepadPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "ToggleFullscreenEvent.hpp"
#include "RebuildBrushEvent.hpp"

#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "ControllerInput.hpp"

#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>

GamepadPublisher::GamepadPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

bool GamepadPublisher::isConnected() const
{
    return glfwJoystickIsGamepad(joystickId);
}

float GamepadPublisher::getLeftStickX() const { return cachedLx; }
float GamepadPublisher::getLeftStickY() const { return cachedLy; }

void GamepadPublisher::pollLeftStick()
{
    if (!glfwJoystickIsGamepad(joystickId)) return;
    GLFWgamepadstate state;
    if (!glfwGetGamepadState(joystickId, &state)) return;

    float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    float mag = std::sqrt(lx * lx + ly * ly);
    if (mag < deadzone) {
        lx = 0.0f;
        ly = 0.0f;
    } else {
        float clampedMag = std::min(mag, 1.0f);
        float scaled = (clampedMag - deadzone) / (1.0f - deadzone);
        float nx = lx / mag;
        float ny = ly / mag;
        lx = nx * scaled;
        ly = ny * scaled;
    }
    cachedLx = lx;
    cachedLy = ly;

    // Cache buttons from the same state read
    cachedA = state.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
    cachedB = state.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
    cachedStart = state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
    cachedBack = state.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
}

bool GamepadPublisher::aButtonPressed()
{
    bool pressed = cachedA && !aPrev;
    aPrev = cachedA;
    return pressed;
}

bool GamepadPublisher::bButtonPressed()
{
    bool pressed = cachedB && !bPrev;
    bPrev = cachedB;
    return pressed;
}

bool GamepadPublisher::menuButtonPressed()
{
    bool pressed = cachedBack && !menuPrev;
    menuPrev = cachedBack;
    return pressed;
}

bool GamepadPublisher::startButtonPressed()
{
    bool pressed = cachedStart && !startPrev;
    startPrev = cachedStart;
    return pressed;
}

void GamepadPublisher::update(EventManager* em, const Camera& cam, float deltaTime, ControllerManager* cm, Brush3dManager* brushManager, bool flipRotation) {
    if (!em || !cm) return;

    // Ensure we have a valid gamepad to poll.
    if (!glfwJoystickIsGamepad(joystickId)) {
        int found = -1;
        for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; ++jid) {
            if (glfwJoystickIsGamepad(jid)) { found = jid; break; }
        }
        if (found == -1) return;
        joystickId = found;
    }

    GLFWgamepadstate state;
    if (!glfwGetGamepadState(joystickId, &state)) return;

    ControllerContext& gctx = cm->gamepadContext;
    const ControllerParameters& cp = *cm->getParameters();

    // Cache buttons from this single state read
    cachedBack = state.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
    cachedA = state.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
    cachedB = state.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
    cachedStart = state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;

    // BACK (View / two squares) -> toggle fullscreen
    if (cachedBack && !backPrev) em->publish(std::make_shared<ToggleFullscreenEvent>());
    backPrev = cachedBack;

    // Cache left stick for radial menu access (circular deadzone + rescale)
    float lxRaw = std::clamp(state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], -1.0f, 1.0f);
    float lyRaw = std::clamp(state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], -1.0f, 1.0f);
    float lmag = std::sqrt(lxRaw * lxRaw + lyRaw * lyRaw);
    if (lmag < deadzone) {
        cachedLx = 0.0f;
        cachedLy = 0.0f;
    } else {
        float clampedMag = std::min(lmag, 1.0f);
        float scaled = (clampedMag - deadzone) / (1.0f - deadzone);
        float nx = lxRaw / lmag;
        float ny = lyRaw / lmag;
        cachedLx = nx * scaled;
        cachedLy = ny * scaled;
    }

    // Local left stick for camera/brush movement (circular deadzone already applied)
    float lx = cachedLx;
    float ly = cachedLy;

    float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    if (std::abs(rx) < deadzone) rx = 0.0f;
    if (std::abs(ry) < deadzone) ry = 0.0f;

    // Use Camera's configured speeds so gamepad feels like keyboard movement.
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    // Triggers -> forward/back translation
    float ltrig = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
    float rtrig = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
    float lval = (ltrig + 1.0f) * 0.5f;
    float rval = (rtrig + 1.0f) * 0.5f;
    float net = (rval - lval);

    // Bumpers -> roll rotation
    bool rollL = state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS;
    bool rollR = state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;

    // ---- Map raw stick/trigger input into an action based on active page ----
    ControllerAction action;
    const PageCategory cat = gctx.activeCategory();
    const PageControl ctrl = gctx.activeControl();

    if (cat == PageCategory::CAMERA) {
        if (lx != 0.0f) action.translate += right * (lx * velocity);
        if (ly != 0.0f) action.translate += up * (-ly * velocity);
        if (net != 0.0f) action.translate += forward * (net * velocity);
        action.rotateDeg.x += rotSign * (-rx * angDeg);
        action.rotateDeg.y += rotSign * (-ry * angDeg);
        if (rollL) action.rotateDeg.z += rotSign * (-angDeg);
        if (rollR) action.rotateDeg.z += rotSign * ( angDeg);
    } else {
        float mSpeed = cp.cameraMoveSpeed * deltaTime;
        float aSpeed = cp.cameraAngularSpeedDeg * deltaTime;
        if (lx != 0.0f) action.translate += right * (lx * mSpeed);
        if (ly != 0.0f) action.translate += up * (-ly * mSpeed);
        if (net != 0.0f) action.translate += forward * (net * mSpeed);
        action.rotateDeg.x += rotSign * (-rx * aSpeed);
        action.rotateDeg.y += rotSign * (-ry * aSpeed);
        if (rollL) action.rotateDeg.z += rotSign * (-aSpeed);
        if (rollR) action.rotateDeg.z += rotSign * ( aSpeed);
        if (lx != 0.0f) action.scaleDelta.x += lx * 0.5f * deltaTime;
        if (ly != 0.0f) action.scaleDelta.y += -ly * 0.5f * deltaTime;
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
