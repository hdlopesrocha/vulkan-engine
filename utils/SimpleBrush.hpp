#include "../math/TexturePainter.hpp"

class SimpleBrush : public TexturePainter {
	int brush;

	public: 
	SimpleBrush(int brush);
	int paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const override;
};

