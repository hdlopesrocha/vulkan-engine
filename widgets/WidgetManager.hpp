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
    void addWidget(std::shared_ptr<Widget> widget) {
        widgets.push_back(widget);
    }
    
    // Render all visible widgets
    void renderAll() {
        for (auto& widget : widgets) {
            if (widget->isVisible()) {
                widget->render();
            }
        }
    }
    
    // Render menu items to toggle widgets
    void renderMenu() {
        if (ImGui::BeginMenu("Windows")) {
            for (auto& widget : widgets) {
                bool isOpen = widget->isVisible();
                if (ImGui::MenuItem(widget->getTitle().c_str(), nullptr, &isOpen)) {
                    if (isOpen) widget->show();
                    else widget->hide();
                }
            }
            ImGui::EndMenu();
        }
    }
    
    // Get widget by title (for programmatic access)
    std::shared_ptr<Widget> getWidget(const std::string& title) {
        for (auto& widget : widgets) {
            if (widget->getTitle() == title) {
                return widget;
            }
        }
        return nullptr;
    }
    
private:
    std::vector<std::shared_ptr<Widget>> widgets;
};
