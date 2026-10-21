#pragma once
#include "Vertex.hpp"
#include "TextureType.hpp"
#include <glm/glm.hpp>
#include <functional>

// Paints a vertex's texture-coordinate technique and UV.
//
// This is deliberately separate from TexturePainter (which chooses the
// brush/material index and HSV): the coordinate technique is a property of the
// shape being tessellated, so every Octree::apply() takes a TextureCoordPainter
// alongside its TexturePainter. The GPU-derived techniques carry UV (0, 0) --
// their coordinates are evaluated in the shader from the world position -- and
// only Custom authors a per-vertex UV.
class TextureCoordPainter {
public:
    virtual ~TextureCoordPainter() = default;

    // The TextureType written to Vertex::textureType.
    virtual TextureType textureType() const = 0;

    // The UV written to Vertex::texCoord. GPU-derived modes return (0, 0).
    virtual glm::vec2 paintUV(const Vertex &v) const { (void)v; return glm::vec2(0.0f); }
};

// Triplanar: weighted X/Y/Z world-position projections computed on the GPU, so
// the vertex carries texCoord = (0, 0) and textureType = Triplanar.
class TriplanarMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::Triplanar; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

// Single-axis world-position projections (GPU).
class ProjectionXMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::ProjectionX; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

class ProjectionYMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::ProjectionY; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

class ProjectionZMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::ProjectionZ; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

// Angular: spherical coordinates around the material's mapping origin (GPU).
class AngularMappingPainter : public TextureCoordPainter {
public:
    TextureType textureType() const override { return TextureType::Angular; }
    glm::vec2 paintUV(const Vertex &) const override { return glm::vec2(0.0f); }
};

// Custom: the UV is authored here (e.g. from the vertex's existing texCoord or a
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
