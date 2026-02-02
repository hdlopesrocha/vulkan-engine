#pragma once
#include <cstddef>

// Simple widget interface for controlling billboard count
#include "Widget.hpp"
class BillboardWidget : public Widget {
public:
    BillboardWidget();
    void setCount(size_t count);
    size_t getCount() const;
    void setScale(float s);
    float getScale() const;
    void render() override;
private:
    size_t count;
    float scale = 1.0f;
};
