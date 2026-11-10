#pragma once

#include "SignedDistanceOperation.hpp"
#include <glm/glm.hpp>

class PaintSignedDistanceOperation final : public SignedDistanceOperation {
public:
    float combine(float existing, float shape) const override {
        // Paint keeps the existing surface where it is already inside the
        // shape: min(existing, max(existing, shape)).
        return glm::min(existing, glm::max(existing, shape));
    }

    bool propagatesFromInfinity() const override { return false; }
    bool preservesSolid() const override { return true; }
    bool preservesEmpty() const override { return true; }
    bool paintsVertices() const override { return true; }
};
