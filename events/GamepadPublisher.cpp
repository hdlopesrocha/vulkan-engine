#include "GamepadPublisher.hpp"
#include "EventManager.hpp"
#include "CameraEvents.hpp"
#include "../vulkan/Camera.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>

GamepadPublisher::GamepadPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

void GamepadPublisher::update(EventManager* em, const Camera& cam, float deltaTime, bool flipRotation) {
    if (!em) return;

    // Use GLFW gamepad API on joystick 1
    if (!glfwJoystickIsGamepad(GLFW_JOYSTICK_1)) return;

    GLFWgamepadstate state;
    if (!glfwGetGamepadState(GLFW_JOYSTICK_1, &state)) return;

    // Read left stick (axes in [-1,1])
    float lx = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float ly = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];

    // Apply deadzone
    if (std::abs(lx) < deadzone) lx = 0.0f;
    if (std::abs(ly) < deadzone) ly = 0.0f;

    float velocity = moveSpeed * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    // Left stick X -> sideways (right is positive)
    if (lx != 0.0f) {
        em->publish(std::make_shared<TranslateCameraEvent>(right * (lx * velocity)));
    }
    // Left stick Y -> up/down. GLFW axis: up is -1, down is +1, so invert to make up positive
    if (ly != 0.0f) {
        em->publish(std::make_shared<TranslateCameraEvent>(up * ((-ly) * velocity)));
    }

    // --- Bumpers: now used for roll rotation (swap with triggers) ---
    if (state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS) {
        // roll left
        float rollDeg = rotSign * (-angularSpeedDeg * deltaTime);
        em->publish(std::make_shared<RotateCameraEvent>(forward, rollDeg));
    }
    if (state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS) {
        // roll right
        float rollDeg = rotSign * (angularSpeedDeg * deltaTime);
        em->publish(std::make_shared<RotateCameraEvent>(forward, rollDeg));
    }

    // --- Rotation using right stick (yaw/pitch) ---
    float rx = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
    float ry = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
    if (std::abs(rx) < deadzone) rx = 0.0f;
    if (std::abs(ry) < deadzone) ry = 0.0f;

    // angular movement in degrees for this frame
    // Flip the right-stick mapping: invert both axes so the analog feels reversed
    float yawDeg = rotSign * (-rx * angularSpeedDeg * deltaTime);
    // invert vertical axis so pushing the stick up results in positive pitch change
    float pitchDeg = rotSign * (-ry * angularSpeedDeg * deltaTime);
    if (yawDeg != 0.0f || pitchDeg != 0.0f) {
        em->publish(std::make_shared<RotateCameraEvent>(yawDeg, pitchDeg, 0.0f));
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
        em->publish(std::make_shared<TranslateCameraEvent>(forward * (net * velocity)));
    }
}
