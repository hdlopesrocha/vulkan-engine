#pragma once

#include <memory>
#include "BillboardWidget.hpp"
#include "../vulkan/VegetationRenderer.hpp"
#include <vector>
#include <glm/glm.hpp>

// Connects the widget and vegetation renderer for real-time billboard control
class BillboardWidgetManager {
public:
    BillboardWidgetManager(std::shared_ptr<BillboardWidget> widget,
                          VegetationRenderer* vegetationRenderer,
                          std::vector<glm::vec3>* allCandidatePositions)
        : widget(widget), vegetationRenderer(vegetationRenderer), allCandidatePositions(allCandidatePositions) {}

    void sync() {
        if (vegetationRenderer && widget && allCandidatePositions) {
            int count = widget->getCount();
            if (count < 0) count = 0;
            if (count > static_cast<int>(allCandidatePositions->size())) count = static_cast<int>(allCandidatePositions->size());
            // Set all visible instances in chunk 0 (or global chunk)
            std::vector<glm::vec3> visible;
            visible.reserve(count);
            for (int i = 0; i < count; ++i) visible.push_back((*allCandidatePositions)[i]);
            vegetationRenderer->setChunkInstances(0, visible);
        }
    }
private:
    std::shared_ptr<BillboardWidget> widget;
    VegetationRenderer* vegetationRenderer;
    std::vector<glm::vec3>* allCandidatePositions;
};
