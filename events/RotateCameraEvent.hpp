#pragma once

#include "Event.hpp"
#include <glm/glm.hpp>



// Event: rotate the camera. Use Euler angles (degrees) or axis/angle depending on handler.
// Here we provide yaw/pitch/roll (degrees) for convenience and also an axis/angle.
class RotateCameraEvent : public Event {
public:
    RotateCameraEvent() = default;
    // yaw, pitch, roll in degrees
    RotateCameraEvent(float yawDeg, float pitchDeg, float rollDeg)
        : yaw(yawDeg), pitch(pitchDeg), roll(rollDeg) {}

    // Euler representation (degrees)
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
};
