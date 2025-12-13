#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include "../events/IEventHandler.hpp"
#include "../events/CameraEvents.hpp"

class Camera : public IEventHandler {
public:
    Camera(const glm::vec3 &pos = glm::vec3(0.0f, 0.0f, 2.5f));

    // Camera is now event-driven. Use events to control translation/rotation.

    // Getters
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const { return position; }
    glm::vec3 getForward() const { return glm::normalize(glm::rotate(orientation, glm::vec3(0.0f, 0.0f, -1.0f))); }
    glm::vec3 getUp() const { return glm::normalize(glm::rotate(orientation, glm::vec3(0.0f, 1.0f, 0.0f))); }
    glm::vec3 getRight() const { return glm::normalize(glm::cross(getForward(), getUp())); }

    // Tunables
    float speed = 2.5f;
    float angularSpeedRad = glm::radians(45.0f);

    // expose orientation for UI/debug
    glm::quat getOrientation() const { return orientation; }

    // Immediate control methods (used by event handlers)
    void translate(const glm::vec3 &d) { position += d; }
    void rotateEuler(float yawDeg, float pitchDeg, float rollDeg);
    void rotateAxisAngle(const glm::vec3 &axis, float angleDeg);

    // IEventHandler implementation: handle camera events
    void onEvent(const EventPtr &event) override;

private:
    glm::vec3 position;
    glm::quat orientation; // camera orientation
};
