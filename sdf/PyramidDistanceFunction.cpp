#include "PyramidDistanceFunction.hpp"


PyramidDistanceFunction::PyramidDistanceFunction() {
    
}

float PyramidDistanceFunction::distance(const glm::vec3 &p, const Transformation &model)  {
   glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;

    // aplicar escala ao ponto, não à geometria
    pos /= model.scale;

    // pirâmide unitária (base half=0.5, altura=1.0)
    float d = SDF::pyramid(pos, 1.0f, sqrt(0.5f));

    // corrigir métrica multiplicando pela menor escala
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

SdfType PyramidDistanceFunction::getType() const {
    return SdfType::PYRAMID;
}

glm::vec3 PyramidDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}
