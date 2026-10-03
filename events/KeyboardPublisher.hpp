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
    KeyboardPublisher() = default;

    // Call each frame to inspect key state and publish zero-or-more events.
    // - window: GLFW window to poll
    // - em: EventManager to publish to (must be valid)
    // - cam: reference to Camera (used only to read forward/right/up)
    // - deltaTime: frame delta in seconds
    void update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, Brush3dManager* brushManager, bool flipRotation);

private:
    // Acceleration timer for translate keys (same exponential ramp as wiimote)
    float translateTimer = 0.0f;

    // edge-tracking for single-action keys
    bool f11Prev = false;
    bool escPrev = false;
    bool spacePrev = false;
};
