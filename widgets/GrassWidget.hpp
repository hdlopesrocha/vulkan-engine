#pragma once

#include "Widget.hpp"

class SdfRenderer;

// Grass-clump editor: the SDF renderer consumes the EXISTING vegetation
// instances (position + vegetation type/biome + smooth normal) as one
// procedural Grass SDF clump per instance, expanding each clump into many
// blades in-shader. Everything here maps 1:1 onto SdfRenderer's grass setters
// (shape edits rebuild the scene on the next frame).
class GrassWidget : public Widget {
public:
    explicit GrassWidget(SdfRenderer* sdfRenderer_);
    void render() override;

private:
    SdfRenderer* sdfRenderer;
};
