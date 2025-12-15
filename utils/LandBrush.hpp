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
	int paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const override;
};