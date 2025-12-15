#ifndef BOUNDING_CUBE_HPP
#define BOUNDING_CUBE_HPP

#include "AbstractBoundingBox.hpp"

class BoundingCube : public AbstractBoundingBox {
public:
    using AbstractBoundingBox::AbstractBoundingBox;
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

    bool overlaps1D(float aMin, float aMax, float bMin, float bMax) const;
    bool overlapsX(const BoundingCube &o) const;
    bool overlapsY(const BoundingCube &o) const;
    bool overlapsZ(const BoundingCube &o) const;
    bool isFaceAdjacent(const BoundingCube &other) const;
    bool isNeighbor(const BoundingCube &o) const;

    bool operator<(const BoundingCube& other) const;
    bool operator==(const BoundingCube& other) const;
};

struct BoundingCubeHasher {
    std::size_t operator()(const BoundingCube &v) const {
        std::size_t hash = 0;
        hash ^= std::hash<glm::vec3>{}(v.getMin()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<float>{}(v.getLengthX()) + 0x01000193 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct BoundingCubeKeyHash {
    size_t operator()(const BoundingCube& key) const {
        size_t h1 = std::hash<int>{}(key.getMinX());
        size_t h2 = std::hash<int>{}(key.getMinY());
        size_t h3 = std::hash<int>{}(key.getMinZ());
        size_t h4 = std::hash<int>{}(key.getLengthX());
        return h1 ^ h2 ^ h3 ^ h4;
    }
};

#endif
