#include "BoundingCube.hpp"
#include "AbstractBoundingBox.hpp"
#include "Math.hpp"
#include "Ray.hpp"

BoundingCube::BoundingCube() : AbstractBoundingBox(glm::vec3(0)) {
	this->length = 0;
}

BoundingCube::BoundingCube(glm::vec3 min_, float length_) : AbstractBoundingBox(min_) {
	this->length = length_;
}

BoundingCube::BoundingCube(const BoundingCube &other) : AbstractBoundingBox(other.getMin()) {
    this->length = other.length;
}

glm::vec3 BoundingCube::getMax() const {
    return getMin()+glm::vec3(length);
}

float BoundingCube::getMaxX() const {
    return getMinX() + length;
}

float BoundingCube::getMaxY() const {
    return getMinY() + length;
}

float BoundingCube::getMaxZ() const {
    return getMinZ() + length;
}

glm::vec3 BoundingCube::getLength() const {
    return glm::vec3(length);
}

float BoundingCube::getLengthX() const {
    return length;
}

float BoundingCube::getLengthY() const {
    return length;
}

float BoundingCube::getLengthZ() const {
    return length;
}

void BoundingCube::setLength(float l) {
    this->length = l;
}

BoundingCube BoundingCube::getChild(int i) const {
    float newLength = 0.5f * length;
    return BoundingCube(min + newLength * glm::vec3(CUBE_CORNERS[i]), newLength);
}

glm::vec3 BoundingCube::getChildCenter(int i) const {
	float newLength = 0.5f*length;
    return (min + 0.5f*newLength) + glm::vec3(CUBE_CORNERS[i])*newLength;
}


bool BoundingCube::intersects(const Ray& ray) const {
    return ray.intersects(*this);
}

void BoundingCube::setMax(glm::vec3 v) {
    this->min = v - glm::vec3(length);
}
