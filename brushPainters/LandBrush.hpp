#pragma once
#include "../math/BrushPainter.hpp"

// Terrain material classifier (height + slope -> material index).
//
// The classification itself runs on the GPU (shaders/includes/LandBrush.glsl)
// per vertex, dispatched by the octree topology compute (OctreeTriangles.comp)
// where the surface position and normal are already available. On the CPU the
// painter only marks the node with LAND_BRUSH_INDEX, and needsPosition() ==
// false lets Octree::shape skip SDF::getPosition / getNormalFromPosition
// entirely for terrain.
class LandBrush : public BrushPainter {
public:
    LandBrush() = default;

    int paint(const OctreeNode &o) const override;
    glm::vec3 paintHSV(const OctreeNode &o) const override;
    bool needsPosition() const override;
};
