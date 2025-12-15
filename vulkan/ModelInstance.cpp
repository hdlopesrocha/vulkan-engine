// Translation unit for ModelInstance wrapper header
#include "ModelInstance.hpp"

ModelInstance::ModelInstance(Model3D* m, const VertexBufferObject& v, const glm::mat4& t, VkDescriptorSet ds, Buffer* ub, VkDescriptorSet sds, Buffer* sub, const MaterialProperties* mat)
    : model(m), vbo(v), transform(t), descriptorSet(ds), uniformBuffer(ub), shadowDescriptorSet(sds), shadowUniformBuffer(sub), material(mat) {}
