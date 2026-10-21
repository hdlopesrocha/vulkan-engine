#pragma once
#include "Vertex.hpp"
#include "TextureType.hpp"
#include <glm/glm.hpp>

// Paints per-vertex surface attributes while the octree is built. The painter
// decides (per node/vertex) which material the surface uses (paint), its HSV
// tint (paintHSV), the texture-coordinate technique (paintTextureType) and the
// matching UV (paintUV).
class TexturePainter {
public:
    virtual int paint(const Vertex &v) const = 0;
    virtual glm::vec3 paintHSV(const Vertex &v) const { return glm::vec3(0.0f, 0.5f, 0.5f); }
    // Texture-coordinate technique for this vertex. Triplanar is the default:
    // its UVs are evaluated on the GPU from the world position, so the default
    // paintUV (0, 0) is the correct companion value.
    virtual TextureType paintTextureType(const Vertex &v) const { return TextureType::Triplanar; }
    // Per-vertex custom UV, used only by TextureType::Custom. Modes that derive
    // their UVs on the GPU (Triplanar / ProjectionX/Y/Z / Angular) ignore it.
    virtual glm::vec2 paintUV(const Vertex &v) const { return glm::vec2(0.0f); }
};
