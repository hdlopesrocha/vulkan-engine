#pragma once

#include "../math/TexturePainter.hpp"

class LandBrush : public TexturePainter {
	int underground;
	int grass;
	int sand;
	int softSand;
	int rock;
	int snow;
	int grassMixSand;
	int grassMixSnow;
	int rockMixGrass;
	int rockMixSnow;
	int rockMixSand;

	public: 
	LandBrush();
	int paint(const Vertex &vertex) const override;
	glm::vec3 paintHSV(const Vertex &vertex) const override { return glm::vec3(0.0f, 0.0f, 1.0f); }
};