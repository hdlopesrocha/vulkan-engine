#pragma once

#include <cstdint>

// Per-layer renderer selection (edited by the LayersWidget). Solid layers
// feed the SolidRenderer's IndirectRenderer, Water layers feed the
// WaterRenderer's one. Both passes still render once per frame; the mapping
// only decides which GPU pipeline a layer's chunks are published into.
enum class LayerRendererType : uint8_t {
    Solid = 0,
    Water = 1,
    Video = 2
};

inline const char* layerRendererName(LayerRendererType r) {
    switch (r) {
        case LayerRendererType::Water: return "Water";
        case LayerRendererType::Video: return "Video";
        default: return "Solid";
    }
}
