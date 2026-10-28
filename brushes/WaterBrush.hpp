#include "../math/TexturePositionPainter.hpp"

class WaterBrush : public TexturePositionPainter {
	int water;

	public: 
	WaterBrush(int water_);
	int paint(const OctreeNode &o) const override;
	glm::vec3 paintHSV(const OctreeNode &o) const override { return glm::vec3(0.0f, 0.5f, 0.5f); }
};