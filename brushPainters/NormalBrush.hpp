#pragma once
#include "../math/BrushPainter.hpp"
#include "../math/BrushMode.hpp"

class NormalBrush : public BrushPainter {
	int brush;
	glm::vec3 hsv;

	public: 
	NormalBrush(int brush_, glm::vec3 hsv_);
	int paint(const OctreeNode &o) const override;
	glm::vec3 paintHSV(const OctreeNode &o) const override;
};

