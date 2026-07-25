#pragma once

#include "SignedDistanceOperation.hpp"
#include <glm/glm.hpp>

class AddSignedDistanceOperation final : public SignedDistanceOperation {
public:
    float combine(float existing, float shape) const override {
        return glm::min(existing, shape);
    }

    bool propagatesFromInfinity() const override { return true; }
    bool preservesSolid() const override { return true; }
    bool preservesEmpty() const override { return false; }
};
