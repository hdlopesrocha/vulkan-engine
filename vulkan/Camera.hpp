#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    Camera(const glm::vec3 &pos = glm::vec3(0.0f, 0.0f, 2.5f));

    // Process keyboard input for this frame
    void processInput(GLFWwindow* window, float deltaTime);

    // Getters
    glm::mat4 getViewMatrix() const;
    glm::vec3 getPosition() const { return position; }
    glm::vec3 getForward() const { return glm::normalize(glm::rotate(orientation, glm::vec3(0.0f, 0.0f, -1.0f))); }
    glm::vec3 getUp() const { return glm::normalize(glm::rotate(orientation, glm::vec3(0.0f, 1.0f, 0.0f))); }
    glm::vec3 getRight() const { return glm::normalize(glm::cross(getForward(), getUp())); }

    // Tunables
    float speed = 2.5f;
    float angularSpeedRad = glm::radians(45.0f);

private:
    glm::vec3 position;
    glm::quat orientation; // camera orientation
};
