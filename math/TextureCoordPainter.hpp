#pragma once
#include "TextureType.hpp"
#include <glm/glm.hpp>
#include <functional>

class OctreeNode;

// Paints a node's texture-coordinate technique, its UV scale and (for Custom)
// its UV, from the octree node itself. Kept separate from TexturePositionPainter
// (which chooses the material + HSV); every Octree::apply() takes both, and
// neither computes the surface position.
//
// The scale is written into the node's `data` -> the vertex's texCoord: for the
// GPU-derived techniques (Triplanar / ProjectionX/Y/Z) it is the (scaleU, scaleV)
// factor applied to the world-space projection in the shader; for Custom it is
// the authored UV. Angular derives its tiling from the material's mappingOrigin,
// so it carries (0, 0).
class TextureCoordPainter {
public:
    virtual ~TextureCoordPainter() = default;

    // The TextureType written to Vertex::textureType.
    virtual TextureType textureType() const = 0;

    // The value written to the node's `data` (the vertex's texCoord).
    virtual glm::vec2 paintUV(const OctreeNode &o) const { (void)o; return glm::vec2(0.0f); }

    // Reserved: whether paintUV() would need the surface position. Octree::shape
    // does not compute one.
    virtual bool needsPosition() const { return false; }
};

// Compatibility umbrella: consumers that include TextureCoordPainter.hpp also
// get the concrete painters (they include this header for the base class).
#include "TriplanarMappingPainter.hpp"
#include "ProjectionXMappingPainter.hpp"
#include "ProjectionYMappingPainter.hpp"
#include "ProjectionZMappingPainter.hpp"
#include "AngularMappingPainter.hpp"
#include "CustomMappingPainter.hpp"
