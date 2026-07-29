#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

class ControllerManager;
class Brush3dManager;

class GamepadPublisher {
public:
    GamepadPublisher(float moveSpeed = 2.5f, float angularSpeedDeg = 45.0f);

    void update(EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, Brush3dManager* brushManager, bool flipRotation);
    void setMoveSpeed(float v) { moveSpeed = v; }
    void setAngularSpeed(float deg) { angularSpeedDeg = deg; }

    bool isConnected() const;
    void pollLeftStick();
    float getLeftStickX() const;
    float getLeftStickY() const;
    bool aButtonPressed();
    bool bButtonPressed();
    bool menuButtonPressed();
    bool startButtonPressed();

private:
    float moveSpeed;
    float angularSpeedDeg;
    int joystickId = GLFW_JOYSTICK_1;

    const float deadzone = 0.15f;

    bool startPrev = false;
    bool backPrev = false;
    bool aPrev = false;
    bool bPrev = false;
    bool menuPrev = false;
    float cachedLx = 0.0f;
    float cachedLy = 0.0f;

    // Cached button states (read once per frame)
    bool cachedA = false;
    bool cachedB = false;
    bool cachedStart = false;
    bool cachedBack = false;
};
