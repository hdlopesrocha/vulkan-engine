#pragma once

#include "SignedDistanceOperation.hpp"
#include <glm/glm.hpp>

class DeleteSignedDistanceOperation final : public SignedDistanceOperation {
public:
    float combine(float existing, float shape) const override {
        return glm::max(existing, -shape);
    }

    bool propagatesFromInfinity() const override { return false; }
    bool preservesSolid() const override { return false; }
    bool preservesEmpty() const override { return true; }
};
