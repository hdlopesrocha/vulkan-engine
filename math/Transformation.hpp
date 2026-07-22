#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Transformation {
public:
    glm::vec3 scale;
    glm::vec3 translate;
    glm::quat quaternion;
    Transformation();
    Transformation(glm::vec3 scale_, glm::vec3 translate_, float yaw, float pitch, float roll);
    Transformation(glm::vec3 scale_, glm::vec3 translate_, const glm::quat& quat_);
    static glm::quat getRotation(float yaw, float pitch, float roll);
};

 
