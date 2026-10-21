#pragma once
#include "../math/TexturePositionPainter.hpp"

class SimpleBrush : public TexturePositionPainter {
	int brush;
	glm::vec3 hsv;

	public: 
	SimpleBrush(int brush_, glm::vec3 hsv_ = glm::vec3(0.0f, 0.5f, 0.5f));
	int paint(const Vertex &vertex) const override;
	glm::vec3 paintHSV(const Vertex &vertex) const override;
};
