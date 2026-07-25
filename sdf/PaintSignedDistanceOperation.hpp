#pragma once

#include "SignedDistanceOperation.hpp"

class PaintSignedDistanceOperation final : public SignedDistanceOperation {
public:
    float combine(float existing, float shape) const override {
        return existing;
    }

    bool propagatesFromInfinity() const override { return false; }
    bool preservesSolid() const override { return true; }
    bool preservesEmpty() const override { return true; }
};
