#include "Transformation.hpp"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

Transformation::Transformation()
    : scale(1.0f, 1.0f, 1.0f), translate(0.0f, 0.0f, 0.0f), quaternion(1.0f, 0.0f, 0.0f, 0.0f)
{
}

Transformation::Transformation(glm::vec3 scale, glm::vec3 translate, float yaw, float pitch, float roll)
    : scale(scale), translate(translate), quaternion(getRotation(yaw, pitch, roll))
{
}

glm::quat Transformation::getRotation(float yaw, float pitch, float roll)
{
    // Build quaternions per axis (angles are in degrees in the interface)
    glm::quat qx = glm::angleAxis(glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat qy = glm::angleAxis(glm::radians(yaw),   glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat qz = glm::angleAxis(glm::radians(roll),  glm::vec3(0.0f, 0.0f, 1.0f));

    // Y * X * Z order (yaw, then pitch, then roll)
    return qy * qx * qz;
}
