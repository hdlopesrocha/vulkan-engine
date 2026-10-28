#pragma once
#include "../math/TexturePositionPainter.hpp"
#include "../math/BrushMode.hpp"

class NormalBrush : public TexturePositionPainter {
	int brush;
	glm::vec3 hsv;
	glm::vec3 normal;

	public: 
	NormalBrush(int brush_);
	NormalBrush(int brush_, glm::vec3 hsv_, glm::vec3 normal_);
	int paint(const OctreeNode &o) const override;
	glm::vec3 paintHSV(const OctreeNode &o) const override;
};

