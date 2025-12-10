#include "Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(const glm::vec3 &pos)
    : position(pos), orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
}

void Camera::processInput(GLFWwindow* window, float deltaTime) {
    if (!window) return;

    // derive axes from quaternion
    glm::vec3 forward = getForward();
    glm::vec3 up = getUp();
    glm::vec3 right = getRight();

    // rotation: FR = yaw (-/+), G/T = pitch (-/+), H/Y = roll (-/+)
    float ang = angularSpeedRad * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(-ang, glm::vec3(0.0f,1.0f,0.0f)) * orientation);
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(ang, glm::vec3(0.0f,1.0f,0.0f)) * orientation);

    // recompute axes after possible yaw change
    forward = getForward();
    up = getUp();
    right = getRight();

    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(-ang, right) * orientation);
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(ang, right) * orientation);

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(-ang, forward) * orientation);
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) orientation = glm::normalize(glm::angleAxis(ang, forward) * orientation);

    // recompute axes after rotation
    forward = getForward();
    up = getUp();
    right = getRight();

    // translation: W/A/S/D forward/left/back/right, Q/E up/down
    float velocity = speed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position += forward * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position -= forward * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position += right * velocity;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) position += up * velocity;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) position -= up * velocity;
}

glm::mat4 Camera::getViewMatrix() const {
    glm::vec3 f = getForward();
    glm::vec3 u = getUp();
    return glm::lookAt(position, position + f, u);
}
