#pragma once
#include "geometry/Vertex.hpp"
#include "TextureType.hpp"
#include <glm/glm.hpp>
#include <functional>

// Paints a vertex's texture-coordinate technique, its UV scale and (for Custom)
// its UV.
//
// This is deliberately separate from TexturePositionPainter (which chooses the
// brush/material index and HSV): the coordinate technique and its scale are a
// property of the shape being tessellated, so every Octree::apply() takes a
// TextureCoordPainter alongside its TexturePositionPainter.
//
// The scale is written into the vertex's texCoord: for the GPU-derived
// techniques (Triplanar / ProjectionX/Y/Z) texCoord is the (scaleU, scaleV)
// factor applied to the world-space projection in the shader; for Custom it is
// the authored UV. Angular derives its tiling from the material's mappingOrigin,
// so it carries (0, 0).
class TextureCoordPainter {
public:
    virtual ~TextureCoordPainter() = default;

    // The TextureType written to Vertex::textureType.
    virtual TextureType textureType() const = 0;

    // The value written to Vertex::texCoord.
    virtual glm::vec2 paintUV(const Vertex &v) const { (void)v; return glm::vec2(0.0f); }
};

// Weighted X/Y/Z world-position projections computed on the GPU; texCoord
// carries the (scaleU, scaleV) applied to the projection.
class TriplanarMappingPainter : public TextureCoordPainter {
public:
    explicit TriplanarMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::Triplanar; }
    glm::vec2 paintUV(const Vertex &) const override { return scale; }
private:
    glm::vec2 scale;
};

// Single-axis world-position projections (GPU); texCoord carries the scale.
class ProjectionXMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionXMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionX; }
    glm::vec2 paintUV(const Vertex &) const override { return scale; }
private:
    glm::vec2 scale;
};

class ProjectionYMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionYMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionY; }
    glm::vec2 paintUV(const Vertex &) const override { return scale; }
private:
    glm::vec2 scale;
};

class ProjectionZMappingPainter : public TextureCoordPainter {
public:
    explicit ProjectionZMappingPainter(glm::vec2 uvScale = glm::vec2(0.01f)) : scale(uvScale) {}
    TextureType textureType() const override { return TextureType::ProjectionZ; }
    glm::vec2 paintUV(const Vertex &) const override { return scale; }
private:
    glm::vec2 scale;
};

// Spherical coordinates around the material's mapping origin (GPU); tiling comes
// from mappingOrigin.w, so the vertex carries (0, 0).
class AngularMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::Angular; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

// The UV is authored here (e.g. from the vertex's existing texCoord or a
// caller-supplied function); the shader samples it directly.
class CustomMappingPainter : public TextureCoordPainter {
public:
    CustomMappingPainter() = default;
    explicit CustomMappingPainter(std::function<glm::vec2(const Vertex &)> fn) : uvFn(std::move(fn)) {}

    TextureType textureType() const override { return TextureType::Custom; }
    glm::vec2 paintUV(const Vertex &v) const override { return uvFn ? uvFn(v) : glm::vec2(0.0f); }

private:
    std::function<glm::vec2(const Vertex &)> uvFn;
};
