#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

// Simple keyboard-to-event publisher. Polls GLFW key state each frame and
// publishes TranslateCameraEvent / RotateCameraEvent / ToggleFullscreenEvent
// / CloseWindowEvent to the provided EventManager. It uses the supplied
// Camera to compute forward/right/up axes for camera-relative motion.
class KeyboardPublisher {
public:
    KeyboardPublisher(float moveSpeed = 2.5f, float angularSpeedDeg = 45.0f);

    // Call each frame to inspect key state and publish zero-or-more events.
    // - window: GLFW window to poll
    // - em: EventManager to publish to (must be valid)
    // - cam: reference to Camera (used only to read forward/right/up)
    // - deltaTime: frame delta in seconds
    void update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime);

private:
    float moveSpeed;         // units per second
    float angularSpeedDeg;   // degrees per second

    // track toggled keys to detect key-down events for single-action keys
    bool f11Prev = false;
    bool escPrev = false;
};
