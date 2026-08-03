#pragma once

#include "SignedDistanceOperation.hpp"
#include "SDF.hpp"

class PaintSignedDistanceOperation final : public SignedDistanceOperation {
public:
    float combine(float existing, float shape) const override {
        return SDF::opPaint(existing, shape);
    }

    bool propagatesFromInfinity() const override { return false; }
    bool preservesSolid() const override { return true; }
    bool preservesEmpty() const override { return true; }
    bool paintsVertices() const override { return true; }
};
