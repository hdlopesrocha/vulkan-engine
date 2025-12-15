#pragma once

#include <string>

// Base class for all UI widgets
class Widget {
public:
    Widget(const std::string& title);
    virtual ~Widget();
    
    // Render the widget (must be implemented by derived classes)
    virtual void render() = 0;
    
    // Widget visibility control
    bool isVisible() const;
    void show();
    void hide();
    void toggle();
    
    // Get widget title
    const std::string& getTitle() const;
    
protected:
    std::string title;
    bool isOpen;
};
