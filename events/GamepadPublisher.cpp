#include "GamepadPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "ToggleFullscreenEvent.hpp"
#include "CloseWindowEvent.hpp"

#include "ControllerManager.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <glm/gtc/constants.hpp>

GamepadPublisher::GamepadPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

void GamepadPublisher::update(EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, bool flipRotation) {
    if (!em) return;

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

    // Read left stick (axes in [-1,1])
    float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];

    // Apply deadzone
    if (std::abs(lx) < deadzone) lx = 0.0f;
    if (std::abs(ly) < deadzone) ly = 0.0f;

    // Button-down toggles: START -> toggle fullscreen, BACK -> close window
    bool startNow = (state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS);
    if (startNow && !startPrev) {
        em->publish(std::make_shared<ToggleFullscreenEvent>());
    }
    startPrev = startNow;

    bool backNow = (state.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS);
    if (backNow && !backPrev) {
        em->publish(std::make_shared<CloseWindowEvent>());
    }
    backPrev = backNow;

    // Use Camera's configured speeds so gamepad feels like keyboard movement
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    ControllerParameters* cp = nullptr;
    if (controllerManager) cp = controllerManager->getParameters();

    // DPAD map: allow direct page selection via DPAD
    if (cp) {
        if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS) cp->currentPage = ControllerParameters::CAMERA;
        if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_POSITION;
        if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_SCALE;
        if (state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_ROTATION;
    }

    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    // Left stick X -> sideways (right is positive)
    if (lx != 0.0f) {
        if (cp && cp->currentPage != ControllerParameters::CAMERA) {
            cp->brushPosition += right * (lx * (cp->cameraMoveSpeed * deltaTime));
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(right * (lx * velocity)));
        }
    }
    // Left stick Y -> up/down (up when stick up). GLFW axis: up is -1, down is +1 so invert to make up positive
    if (ly != 0.0f) {
        if (cp && cp->currentPage != ControllerParameters::CAMERA) {
            cp->brushPosition += up * ((-ly) * (cp->cameraMoveSpeed * deltaTime));
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(up * ((-ly) * velocity)));
        }
    }

    // --- Bumpers: now used for roll rotation (swap with triggers) ---
    if (state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            cp->brushRotation.z -= rotSign * angDeg;
        } else {
            // roll left
            float rollDeg = rotSign * (-angDeg);
            em->publish(std::make_shared<RotateCameraEvent>(forward, rollDeg));
        }
    }
    if (state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            cp->brushRotation.z += rotSign * angDeg;
        } else {
            // roll right
            float rollDeg = rotSign * (angDeg);
            em->publish(std::make_shared<RotateCameraEvent>(forward, rollDeg));
        }
    }

    // --- Rotation using right stick (yaw/pitch) ---
    float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    if (std::abs(rx) < deadzone) rx = 0.0f;
    if (std::abs(ry) < deadzone) ry = 0.0f;

    // angular movement in degrees for this frame
    // Flip the right-stick mapping: invert both axes so the analog feels reversed
    float yawDeg = rotSign * (-rx * angDeg);
    // invert vertical axis so pushing the stick up results in positive pitch change
    float pitchDeg = rotSign * (-ry * angDeg);
    if (yawDeg != 0.0f || pitchDeg != 0.0f) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            cp->brushRotation.y += yawDeg;
            cp->brushRotation.x += pitchDeg;
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(yawDeg, pitchDeg, 0.0f));
        }
    }

    // --- Forward/back translation using analog triggers (swap with bumpers) ---
    // GLFW trigger axes typically range from -1 (released) to 1 (pressed) on many controllers.
    float ltrig = state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
    float rtrig = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
    // map to 0..1
    float lval = (ltrig + 1.0f) * 0.5f;
    float rval = (rtrig + 1.0f) * 0.5f;
    // net forward amount: right trigger -> forward, left trigger -> backward
    float net = (rval - lval); // in [-1,1]
    if (std::abs(net) > 1e-4f) {
        if (cp && cp->currentPage != ControllerParameters::CAMERA) {
            cp->brushPosition += forward * (net * (cp->cameraMoveSpeed * deltaTime));
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(forward * (net * velocity)));
        }
    }
}
