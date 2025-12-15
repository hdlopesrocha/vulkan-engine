#pragma once
#include "AbstractBoundingBox.hpp"

class BoundingBox : public AbstractBoundingBox {
public:
    using AbstractBoundingBox::AbstractBoundingBox;
private:
    glm::vec3 max;
public:
    BoundingBox(glm::vec3 min, glm::vec3 max);
    BoundingBox();
    void setMax(glm::vec3 v);
    void setMaxX(float v);
    void setMaxY(float v);
    void setMaxZ(float v);
    void setMinX(float v);
    void setMinY(float v);
    void setMinZ(float v);

    glm::vec3 getLength() const override;
    float getLengthX() const override;
    float getLengthY() const override;
    float getLengthZ() const override;
    float getMaxX() const override;
    float getMaxY() const override;
    float getMaxZ() const override;
    glm::vec3 getMax() const override;
};

 
