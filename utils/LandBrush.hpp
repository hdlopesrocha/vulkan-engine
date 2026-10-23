#pragma once
#include "../math/TexturePositionPainter.hpp"
#include "../math/BrushMode.hpp"

// Terrain material classifier (height + slope -> material index).
//
// The classification itself runs on the GPU (shaders/includes/LandBrush.glsl)
// per vertex, dispatched by the octree topology compute (OctreeTriangles.comp)
// where the surface position and normal are already available. On the CPU the
// painter only marks the node with LAND_BRUSH_INDEX, and needsPosition() ==
// false lets Octree::shape skip SDF::getPosition / getNormalFromPosition
// entirely for terrain.
class LandBrush : public TexturePositionPainter {
public:
    LandBrush() = default;

    int paint(const OctreeNode &o) const override { (void)o; return LAND_BRUSH_INDEX; }
    glm::vec3 paintHSV(const OctreeNode &o) const override { (void)o; return glm::vec3(0.0f, 0.5f, 0.5f); }
    bool needsPosition() const override { return false; }
};
