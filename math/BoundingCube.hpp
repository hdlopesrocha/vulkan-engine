#pragma once
#include "AbstractBoundingBox.hpp"

class Ray;

class BoundingCube : public AbstractBoundingBox {
public:
    using AbstractBoundingBox::AbstractBoundingBox;
    using AbstractBoundingBox::intersects;
protected:
    float length;
public:
    BoundingCube();
    BoundingCube(glm::vec3 min, float length);
    BoundingCube(const BoundingCube &other);
    void setLength(float l);
    void setMinX(float v);
    void setMinY(float v);
    void setMinZ(float v);
    void setMaxX(float v);
    void setMaxY(float v);
    void setMaxZ(float v);
    void setMax(glm::vec3 v);

    glm::vec3 getLength() const override;
    float getLengthX() const override;
    float getLengthY() const override;
    float getLengthZ() const override;
    float getMaxX() const override;
    float getMaxY() const override;
    float getMaxZ() const override;
    glm::vec3 getMax() const override;

    BoundingCube getChild(int i) const;
    glm::vec3 getChildCenter(int i) const;

    // Non-virtual child-index lookup (avoids 4 virtual getMin*/getLengthX calls).
    // Value-preserving wrt the free getNodeIndex(): BoundingCube stores min/length
    // directly and getLengthX()/getMin*() merely return them.
    inline int getChildIndex(const glm::vec3 &vec) const {
        const float half = length * 0.5f;
        return (vec.x >= min.x + half ? 4 : 0)
             + (vec.y >= min.y + half ? 2 : 0)
             + (vec.z >= min.z + half ? 1 : 0);
    }

    bool overlaps1D(float aMin, float aMax, float bMin, float bMax) const;
    bool overlapsX(const BoundingCube &o) const;
    bool overlapsY(const BoundingCube &o) const;
    bool overlapsZ(const BoundingCube &o) const;
    bool isFaceAdjacent(const BoundingCube &other) const;
    bool isNeighbor(const BoundingCube &o) const;
    bool intersects(const Ray& ray) const;

    bool operator<(const BoundingCube& other) const;
    bool operator==(const BoundingCube& other) const;
};



 
