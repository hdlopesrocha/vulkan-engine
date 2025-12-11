#pragma once

#include "Event.hpp"
#include <glm/glm.hpp>
#include <string>

// Event: translate the camera by a delta vector (world or local-space as agreed by handler)
class TranslateCameraEvent : public Event {
public:
    TranslateCameraEvent() = default;
    TranslateCameraEvent(const glm::vec3 &d) : delta(d) {}
    explicit TranslateCameraEvent(float x, float y, float z) : delta(x,y,z) {}

    std::string name() const override { return "TranslateCameraEvent"; }

    glm::vec3 delta{0.0f, 0.0f, 0.0f};
};

// Event: rotate the camera. Use Euler angles (degrees) or axis/angle depending on handler.
// Here we provide yaw/pitch/roll (degrees) for convenience and also an axis/angle.
class RotateCameraEvent : public Event {
public:
    RotateCameraEvent() = default;
    // yaw, pitch, roll in degrees
    RotateCameraEvent(float yawDeg, float pitchDeg, float rollDeg)
        : yaw(yawDeg), pitch(pitchDeg), roll(rollDeg) {}

    // axis-angle constructor (angle in degrees)
    RotateCameraEvent(const glm::vec3 &axis_, float angleDeg)
        : axis(axis_), angleDegrees(angleDeg), useAxisAngle(true) {}

    std::string name() const override { return "RotateCameraEvent"; }

    // Euler representation (degrees)
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;

    // Axis-angle representation (used if useAxisAngle == true)
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    float angleDegrees = 0.0f;
    bool useAxisAngle = false;
};
