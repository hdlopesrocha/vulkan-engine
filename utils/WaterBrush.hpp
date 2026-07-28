#include "../math/TexturePainter.hpp"

class WaterBrush : public TexturePainter {
	int water;

	public: 
	WaterBrush(int water_);
	int paint(const Vertex &vertex) const override;
	glm::vec3 paintHSV(const Vertex &vertex) const override { return glm::vec3(0.0f, 0.5f, 0.5f); }
};