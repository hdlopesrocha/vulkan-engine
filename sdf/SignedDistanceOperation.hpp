#pragma once

class SignedDistanceOperation {
public:
    virtual ~SignedDistanceOperation() = default;
    virtual float combine(float existing, float shape) const = 0;
    virtual bool propagatesFromInfinity() const = 0;
    virtual bool preservesSolid() const = 0;
    virtual bool preservesEmpty() const = 0;
};
