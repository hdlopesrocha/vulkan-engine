#pragma once

#include "Widget.hpp"

class SdfRenderer;

// Rock-boulder editor: the SDF renderer collects boulder candidates from
// brush-7 (rock) terrain chunks at one per `minSpacing` m^2 and rebuilds the
// scene whenever a control changes, decimating the retained candidate set to
// the live spacing. Everything here maps 1:1 onto SdfRenderer's rock setters,
// so edits are applied on the next frame (real time).
class RocksWidget : public Widget {
public:
    explicit RocksWidget(SdfRenderer* sdfRenderer_);
    void render() override;

private:
    SdfRenderer* sdfRenderer;
};
