#pragma once

#include "Widget.hpp"

class VegetationRenderer;

// Bladed-grass editor: triangle tufts (procedural color, wind) + captured
// impostors for the far field. Everything here maps onto VegetationRenderer
// (tuft scale, impostor hand-off); wind itself lives in the Wind widget.
class GrassWidget : public Widget {
public:
    explicit GrassWidget(VegetationRenderer* vegRenderer_);
    void render() override;

private:
    VegetationRenderer* vegRenderer;
};
