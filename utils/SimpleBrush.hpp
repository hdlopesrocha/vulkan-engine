#pragma once
#include "../math/TexturePainter.hpp"

class SimpleBrush : public TexturePainter {
	int brush;
	glm::vec3 hsv;
	TextureType textureType;

	public: 
	SimpleBrush(int brush_, glm::vec3 hsv_ = glm::vec3(0.0f, 0.5f, 0.5f),
	            TextureType textureType_ = TextureType::Triplanar);
	int paint(const Vertex &vertex) const override;
	glm::vec3 paintHSV(const Vertex &vertex) const override;
	TextureType paintTextureType(const Vertex &vertex) const override;
};
