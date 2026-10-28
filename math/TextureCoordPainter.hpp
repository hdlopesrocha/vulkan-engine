#pragma once
#include "TextureType.hpp"
#include <glm/glm.hpp>

class OctreeNode;

// Paints a node's texture-coordinate technique, its UV scale and (for Custom)
// its UV, from the octree node itself. Kept separate from BrushPainter
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
    virtual ~TextureCoordPainter();

    // The TextureType written to Vertex::textureType.
    virtual TextureType textureType() const = 0;

    // The value written to the node's `data` (the vertex's texCoord).
    virtual glm::vec2 paintUV(const OctreeNode &o) const;

    // Reserved: whether paintUV() would need the surface position. Octree::shape
    // does not compute one.
    virtual bool needsPosition() const;
};

// Compatibility umbrella: consumers that include TextureCoordPainter.hpp also
// get the concrete painters (they include this header for the base class).
#include "../textureCoordPainters/TriplanarMappingPainter.hpp"
#include "../textureCoordPainters/ProjectionXMappingPainter.hpp"
#include "../textureCoordPainters/ProjectionYMappingPainter.hpp"
#include "../textureCoordPainters/ProjectionZMappingPainter.hpp"
#include "../textureCoordPainters/AngularMappingPainter.hpp"
#include "../textureCoordPainters/CustomMappingPainter.hpp"
