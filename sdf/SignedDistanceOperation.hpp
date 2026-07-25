#pragma once

#include "SDF.hpp"

class SignedDistanceOperation {
public:
    using Func = float (*)(float, float);

    explicit SignedDistanceOperation(Func func)
        : m_func(func)
    {
        m_propagatesFromInfinity = (func == SDF::opUnion || func == SDF::opXor);
        m_preservesSolid = (func == SDF::opUnion || func == SDF::opPaint);
        m_preservesEmpty = (func == SDF::opIntersection ||
                            func == SDF::opSubtraction ||
                            func == SDF::opPaint);
    }

    float combine(float existing, float shape) const {
        return m_func(existing, shape);
    }

    bool propagatesFromInfinity() const { return m_propagatesFromInfinity; }
    bool preservesSolid() const { return m_preservesSolid; }
    bool preservesEmpty() const { return m_preservesEmpty; }
    Func raw() const { return m_func; }

private:
    Func m_func;
    bool m_propagatesFromInfinity;
    bool m_preservesSolid;
    bool m_preservesEmpty;
};
