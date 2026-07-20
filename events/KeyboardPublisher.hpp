#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

// Simple keyboard-to-event publisher. Polls GLFW key state each frame and
// publishes TranslateCameraEvent / RotateCameraEvent / ToggleFullscreenEvent
// / CloseWindowEvent to the provided EventManager. It uses the supplied
// Camera to compute forward/right/up axes for camera-relative motion.
class ControllerManager;
class Brush3dManager;

class KeyboardPublisher {
public:
    KeyboardPublisher(float moveSpeed = 2.5f, float angularSpeedDeg = 45.0f);

    // Call each frame to inspect key state and publish zero-or-more events.
    // - window: GLFW window to poll
    // - em: EventManager to publish to (must be valid)
    // - cam: reference to Camera (used only to read forward/right/up)
    // - deltaTime: frame delta in seconds
    void update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, Brush3dManager* brushManager, bool flipRotation);
    void setMoveSpeed(float v) { moveSpeed = v; }
    void setAngularSpeed(float deg) { angularSpeedDeg = deg; }

private:
    float moveSpeed;         // units per second
    float angularSpeedDeg;   // degrees per second

    // edge-tracking for single-action keys
    bool f11Prev = false;
    bool escPrev = false;
    bool k1Prev = false, k2Prev = false, k3Prev = false, k4Prev = false;
    bool k5Prev = false, k6Prev = false, k7Prev = false, k8Prev = false;
};
