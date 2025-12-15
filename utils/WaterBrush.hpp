#include "../math/TexturePainter.hpp"

class WaterBrush : public TexturePainter {
	int water;

	public: 
	WaterBrush(int water);
	int paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const override;
};