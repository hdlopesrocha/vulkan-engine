#pragma once

#include "Widget.hpp"
#include <vector>
#include <memory>
#include <imgui.h>

// Manages all UI widgets and provides menu bar integration
class WidgetManager {
public:
    WidgetManager() = default;

    // Register a widget
    void addWidget(std::shared_ptr<Widget> widget);

    // Render all visible widgets
    void renderAll();

    // Render all visible widgets except the listed instances. Used while the
    // background content build owns the octree: the explorer walks raw nodes
    // and must not run concurrently.
    void renderAllExcept(const std::vector<Widget*>& skip);

    // Render menu items to toggle widgets
    void renderMenu();

private:
    std::vector<std::shared_ptr<Widget>> widgets;
};
