#include "HeightMap.hpp"
#include "HeightFunction.hpp"
#include "Math.hpp"

HeightMap::HeightMap(const HeightFunction &func_, BoundingBox box, float step_)  : BoundingBox(box), step(step_), func(func_){

}

float HeightMap::distance(const glm::vec3 p) const {
    float h = func.getHeightAt(p.x, p.z);

    float h_xp = func.getHeightAt(p.x + step, p.z);
    float h_xm = func.getHeightAt(p.x - step, p.z);
    float h_zp = func.getHeightAt(p.x, p.z + step);
    float h_zm = func.getHeightAt(p.x, p.z - step);

    float dhdx = (h_xp - h_xm) / (2.0f * step);
    float dhdz = (h_zp - h_zm) / (2.0f * step);

    glm::vec3 normal = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));

    glm::vec3 delta = p - glm::vec3(p.x, h, p.z);

    return glm::dot(delta, normal);
}
