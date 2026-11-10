#pragma once

#include <glm/glm.hpp>
#include "../math/BrushMode.hpp"

// Material state a parent cell inherits from its painted surface children.
// `brush` is DISCARD_BRUSH_INDEX when no surface child carries a real brush;
// the remaining fields are those of the child that supplied the brush.
struct BrushTally {
    int brush = DISCARD_BRUSH_INDEX;
    glm::vec3 hsv = glm::vec3(0.0f, 0.5f, 0.5f);
    int textureType = 0;
    glm::vec2 data = glm::vec2(0.0f);
};

// Result of attempting to simplify eight child nodes into their parent.
// `isSimplified` indicates whether simplification succeeded. `brushTally` is
// always filled (even when simplification is rejected): the parent's material
// aggregation is independent of the collapse decision.
struct SimplificationResult {
    bool isSimplified;
    BrushTally brushTally;
    SimplificationResult(bool simplified)
        : isSimplified(simplified) {
    }
};
