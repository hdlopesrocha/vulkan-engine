#pragma once

class SignedDistanceOperation {
public:
    virtual ~SignedDistanceOperation() = default;
    virtual float combine(float existing, float shape) const = 0;
    virtual bool propagatesFromInfinity() const = 0;
    virtual bool preservesSolid() const = 0;
    virtual bool preservesEmpty() const = 0;
    // Operations that apply the painter's material/hsv to existing nodes
    // (e.g. Paint) must still reach pruned subtrees; false for pure
    // geometry ops like Add/Delete.
    virtual bool paintsVertices() const { return false; }
};
