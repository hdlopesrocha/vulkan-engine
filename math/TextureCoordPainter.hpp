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

// Weighted X/Y/Z world-position projections computed on the GPU; data carries
// the (scaleU, scaleV) applied to the projection.
class TriplanarMappingPainter : public TextureCoordPainter {
public:
    explicit TriplanarMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::Triplanar; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};

// Single-axis world-position projections (GPU); data carries the scale.
class ProjectionXMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionXMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionX; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};

class ProjectionYMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionYMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionY; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};

class ProjectionZMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionZMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionZ; }
    glm::vec2 paintUV(const OctreeNode &) const override { return scale; }
private:
    glm::vec2 scale;
};

// Spherical coordinates around the material's mapping origin (GPU); tiling comes
// from mappingOrigin.w, so data carries (0, 0).
class AngularMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::Angular; }
    glm::vec2 paintUV(const OctreeNode &) const override { return glm::vec2(0.0f); }
};

// The UV is authored here (a caller-supplied function over the node); the shader
// samples it directly.
class CustomMappingPainter : public TextureCoordPainter {
public:
    CustomMappingPainter() = default;
    explicit CustomMappingPainter(std::function<glm::vec2(const OctreeNode &)> fn) : uvFn(std::move(fn)) {}

    TextureType textureType() const override { return TextureType::Custom; }
    glm::vec2 paintUV(const OctreeNode &o) const override { return uvFn ? uvFn(o) : glm::vec2(0.0f); }

private:
    std::function<glm::vec2(const OctreeNode &)> uvFn;
};
