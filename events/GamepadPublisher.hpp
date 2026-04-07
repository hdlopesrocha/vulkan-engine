#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

// Polls the first connected gamepad using GLFW's gamepad API and publishes
// translation events according to left stick and shoulder bumpers:
// - Left stick X -> translate sideways (right/left)
// - Left stick Y -> translate up/down (up when stick up)
// - Left bumper (L1) -> translate backward
// - Right bumper (R1) -> translate forward
class GamepadPublisher {
public:
    GamepadPublisher(float moveSpeed = 2.5f, float angularSpeedDeg = 45.0f);

    // Call each frame to poll gamepad and publish events via EventManager.
    // - em: EventManager to publish to
    // - cam: reference to Camera for axis vectors
    // - deltaTime: frame delta seconds
    void update(EventManager* em, const Camera& cam, float deltaTime, bool flipRotation);
    void setMoveSpeed(float v) { moveSpeed = v; }
    void setAngularSpeed(float deg) { angularSpeedDeg = deg; }

private:
    float moveSpeed;
    float angularSpeedDeg; // degrees per second for right stick / triggers
    int joystickId = GLFW_JOYSTICK_1; // current joystick id to poll (defaults to GLFW_JOYSTICK_1)

    // deadzone for thumbstick
    const float deadzone = 0.15f;

    // track toggled buttons to detect button-down events for single-action buttons
    bool startPrev = false;
    bool backPrev = false;
};
