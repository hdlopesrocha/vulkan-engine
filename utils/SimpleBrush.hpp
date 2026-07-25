#pragma once
#include "../math/TexturePainter.hpp"

class SimpleBrush : public TexturePainter {
	int brush;

	public: 
	SimpleBrush(int brush_);
	int paint(const Vertex &vertex) const override;
};

