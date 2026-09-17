#pragma once
#include "../math/TexturePainter.hpp"
#include "../math/BrushMode.hpp"

class NormalBrush : public TexturePainter {
	int brush;
	glm::vec3 hsv;
	glm::vec3 normal;

	public: 
	NormalBrush(int brush_);
	NormalBrush(int brush_, glm::vec3 hsv_, glm::vec3 normal_);
	int paint(const Vertex &vertex) const override;
	glm::vec3 paintHSV(const Vertex &vertex) const override;
};

