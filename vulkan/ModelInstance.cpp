// Translation unit for ModelInstance wrapper header
#include "ModelInstance.hpp"

ModelInstance::ModelInstance(Mesh3D* m, const VertexBufferObject& v, const glm::mat4& t, VkDescriptorSet ds, VkDescriptorSet sds)
    : model(m), vbo(v), transform(t), descriptorSet(ds), shadowDescriptorSet(sds) {}
