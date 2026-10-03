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

    // Render menu items to toggle widgets
    void renderMenu();

private:
    std::vector<std::shared_ptr<Widget>> widgets;
};
