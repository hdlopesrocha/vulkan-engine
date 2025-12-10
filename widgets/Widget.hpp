#pragma once

#include <string>

// Base class for all UI widgets
class Widget {
public:
    Widget(const std::string& title) : title(title), isOpen(false) {}
    virtual ~Widget() = default;
    
    // Render the widget (must be implemented by derived classes)
    virtual void render() = 0;
    
    // Widget visibility control
    bool isVisible() const { return isOpen; }
    void show() { isOpen = true; }
    void hide() { isOpen = false; }
    void toggle() { isOpen = !isOpen; }
    
    // Get widget title
    const std::string& getTitle() const { return title; }
    
protected:
    std::string title;
    bool isOpen;
};
