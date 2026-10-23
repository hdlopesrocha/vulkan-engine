#pragma once
#include <glm/glm.hpp>

class OctreeNode;

// Paints a node's material assignment. The painter receives the octree node
// itself (its current brushIndex / hsv / textureType / data), not a surface
// Vertex — Octree::shape never computes SDF::getPosition / getNormalFromPosition
// for it.
class TexturePositionPainter {
public:
    virtual int paint(const OctreeNode &o) const = 0;
    virtual glm::vec3 paintHSV(const OctreeNode &o) const { return glm::vec3(0.0f, 0.5f, 0.5f); }
    // Reserved: whether paint()/paintHSV() would need the surface position.
    // Octree::shape does not compute one.
    virtual bool needsPosition() const { return true; }
};
