#ifndef SPACE_INSTANCE_DATA_HPP
#define SPACE_INSTANCE_DATA_HPP

#include <glm/glm.hpp>
typedef unsigned int uint;

struct alignas(16) InstanceData {
public:
	glm::mat4 matrix;
	float shift;
	uint animation;
	InstanceData();
	InstanceData(uint animation, const glm::mat4 &matrix, float shift);
};

#endif