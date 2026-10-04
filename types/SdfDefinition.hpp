#pragma once

// SDF definition: which primitive, how it combines, its packed parameters.
// Local-space evaluation; instances carry the world transform.
#include "types/SdfPrimitiveType.hpp"
#include "types/SdfOpType.hpp"
#include <glm/glm.hpp>
#include <cstdint>

namespace sdf_gpu {

struct Definition {
    SdfPrimitiveType prim = SdfPrimitiveType::Sphere;
    SdfOpType op = SdfOpType::Union;
    glm::vec4 params0 = glm::vec4(0.0f);
    glm::vec4 params1 = glm::vec4(0.0f);
    uint32_t deformFlags = 0;
    float smoothK = 0.5f;
};

} // namespace sdf_gpu
