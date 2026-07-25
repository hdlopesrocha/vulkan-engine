#include "../math/TexturePainter.hpp"

class WaterBrush : public TexturePainter {
	int water;

	public: 
	WaterBrush(int water_);
	int paint(const Vertex &vertex) const override;
};