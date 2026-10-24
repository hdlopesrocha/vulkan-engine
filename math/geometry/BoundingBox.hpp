#pragma once
#include "AbstractBoundingBox.hpp"

class BoundingBox : public AbstractBoundingBox {
public:
    using AbstractBoundingBox::AbstractBoundingBox;
private:
    glm::vec3 max;
public:
    BoundingBox(glm::vec3 min_, glm::vec3 max_);
    BoundingBox();
    void setMax(glm::vec3 v);
    void setMaxY(float v);
    void setMinY(float v);

    glm::vec3 getLength() const override;
    float getLengthX() const override;
    float getLengthY() const override;
    float getLengthZ() const override;
    float getMaxX() const override;
    float getMaxY() const override;
    float getMaxZ() const override;
    glm::vec3 getMax() const override;
};

 
