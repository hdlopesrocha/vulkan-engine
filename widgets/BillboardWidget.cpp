#include "BillboardWidget.hpp"
#ifdef USE_IMGUI
#include <imgui.h>
#endif


BillboardWidget::BillboardWidget() : Widget("Billboard Count"), count(100), scale(1.0f) {}
void BillboardWidget::setScale(float s) {
    scale = s;
}

float BillboardWidget::getScale() const {
    return scale;
}

void BillboardWidget::setCount(size_t c) {
    count = c;
}

size_t BillboardWidget::getCount() const {
    return count;
}

void BillboardWidget::render() {
#ifdef USE_IMGUI
    ImGui::SliderInt("Billboard Count", (int*)&count, 0, 1000);
    ImGui::SliderFloat("Billboard Scale", &scale, 0.1f, 10.0f, "%.2f");
#endif
}
