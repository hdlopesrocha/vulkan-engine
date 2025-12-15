#include "Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include "../events/TranslateCameraEvent.hpp"
#include "../events/RotateCameraEvent.hpp"

Camera::Camera(const glm::vec3 &pos)
    : position(pos), orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
}

// Camera input is now event-driven. Keyboard handling is performed by a separate
// KeyboardPublisher which converts key state into camera/window events and
// publishes them to the EventManager.

glm::mat4 Camera::getViewMatrix() const {
    glm::vec3 f = getForward();
    glm::vec3 u = getUp();
    return glm::lookAt(position, position + f, u);
}

void Camera::rotateEuler(float yawDeg, float pitchDeg, float rollDeg) {
    // Convert degrees to radians and apply yaw (around world up), pitch (around right), roll (around forward)
    float yaw = glm::radians(yawDeg);
    float pitch = glm::radians(pitchDeg);
    float roll = glm::radians(rollDeg);

    // Apply yaw around world up (0,1,0)
    orientation = glm::normalize(glm::angleAxis(yaw, glm::vec3(0.0f,1.0f,0.0f)) * orientation);

    // Apply pitch around camera right
    glm::vec3 right = getRight();
    orientation = glm::normalize(glm::angleAxis(pitch, right) * orientation);

    // Apply roll around camera forward
    glm::vec3 forward = getForward();
    orientation = glm::normalize(glm::angleAxis(roll, forward) * orientation);
}

void Camera::rotateAxisAngle(const glm::vec3 &axis, float angleDeg) {
    float angle = glm::radians(angleDeg);
    orientation = glm::normalize(glm::angleAxis(angle, glm::normalize(axis)) * orientation);
}

void Camera::onEvent(const EventPtr &event) {
    if (!event) return;
    // Translate
    if (auto t = std::dynamic_pointer_cast<TranslateCameraEvent>(event)) {
        translate(t->delta);
        return;
    }
    // Rotate
    if (auto r = std::dynamic_pointer_cast<RotateCameraEvent>(event)) {
        if (r->useAxisAngle) rotateAxisAngle(r->axis, r->angleDegrees);
        else rotateEuler(r->yaw, r->pitch, r->roll);
        return;
    }
}

// Moved from header: immediate translation
void Camera::translate(const glm::vec3 &d) {
    position += d;
}
