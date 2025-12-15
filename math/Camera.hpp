#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

class Camera {
public:
    glm::mat4 projection;
    glm::mat4 view;
    glm::quat quaternion;
    glm::vec3 position;
    float near;
    float far;
    float rotationSensitivity;
    float translationSensitivity;

    Camera(glm::vec3 position, glm::quat quaternion, float near, float far);
    glm::vec3 getCameraDirection();
    glm::mat4 getViewProjection();
};

#endif
