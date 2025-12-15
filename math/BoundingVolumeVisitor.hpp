// Auto-generated wrapper header for BoundingVolumeVisitor
#pragma once

#include "BoundingVolume.hpp"

class BoundingVolumeVisitor {
public:
    virtual void visit(const AbstractBoundingBox &box) = 0;
    virtual void visit(const BoundingSphere &sphere) = 0;
    virtual void visit(const BoundingCube &cube) = 0;
    virtual ~BoundingVolumeVisitor() = default;
};
