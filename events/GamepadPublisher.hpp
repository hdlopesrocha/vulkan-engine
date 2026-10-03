#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

class ControllerManager;
class Brush3dManager;

class GamepadPublisher {
public:
    GamepadPublisher() = default;

    void update(EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, Brush3dManager* brushManager, bool flipRotation);

    bool isConnected() const;
    void pollLeftStick();
    float getLeftStickX() const;
    float getLeftStickY() const;
    bool aButtonPressed();
    bool bButtonPressed();
    bool startButtonPressed();

private:
    int joystickId = GLFW_JOYSTICK_1;

    const float deadzone = 0.15f;

    bool startPrev = false;
    bool backPrev = false;
    bool aPrev = false;
    bool bPrev = false;
    float cachedLx = 0.0f;
    float cachedLy = 0.0f;

    // Cached button states (read once per frame)
    bool cachedA = false;
    bool cachedB = false;
    bool cachedStart = false;
    bool cachedBack = false;

    // Translation acceleration (matches keyboard/nunchuk exponential ramp)
    float translateTimer = 0.0f;
};
