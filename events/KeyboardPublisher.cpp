#include "KeyboardPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "CloseWindowEvent.hpp"
#include "ToggleFullscreenEvent.hpp"

KeyboardPublisher::KeyboardPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

void KeyboardPublisher::update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime, bool flipRotation) {
    if (!window || !em) return;

    // compute camera axes (camera provides normalized vectors)
    glm::vec3 forward = cam.getForward();
    glm::vec3 up = cam.getUp();
    glm::vec3 right = cam.getRight();

    // Use Camera's configured speeds so UI changes in CameraWidget take effect.
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    // Translation keys (continuous while held)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(forward * velocity));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(-forward * velocity));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(-right * velocity));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(right * velocity));
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(up * velocity));
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        em->publish(std::make_shared<TranslateCameraEvent>(-up * velocity));
    }

    // Rotation keys (continuous while held)
    // yaw: H (left), F (right) around world up
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(rotSign * -angDeg, 0.0f, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(rotSign * angDeg, 0.0f, 0.0f));
    }
    // pitch: G (down), T (up) around camera right
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(0.0f, rotSign * -angDeg, 0.0f));
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(0.0f, rotSign * angDeg, 0.0f));
    }
    // roll: R (left), Y (right) around forward
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(0.0f, 0.0f, rotSign * -angDeg));
    }
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
        em->publish(std::make_shared<RotateCameraEvent>(0.0f, 0.0f, rotSign * angDeg));
    }

    // Toggle fullscreen: F11 on key-down
    bool f11Now = (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS);
    if (f11Now && !f11Prev) {
        em->publish(std::make_shared<ToggleFullscreenEvent>());
    }
    f11Prev = f11Now;

    // Close window: ESC on key-down
    bool escNow = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
    if (escNow && !escPrev) {
        em->publish(std::make_shared<CloseWindowEvent>());
    }
    escPrev = escNow;
}
