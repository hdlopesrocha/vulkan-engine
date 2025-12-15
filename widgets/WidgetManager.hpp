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

    // Get widget by title (for programmatic access)
    std::shared_ptr<Widget> getWidget(const std::string& title);
    
private:
    std::vector<std::shared_ptr<Widget>> widgets;
};
