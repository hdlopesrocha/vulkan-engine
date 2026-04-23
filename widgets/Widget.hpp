#pragma once

#include <string>

// Base class for all UI widgets
class Widget {
public:
    Widget(const std::string& title, const std::string& icon = {});
    // Accept UTF-8 char8_t string literals (u8"...") on C++20 compilers
    Widget(const std::string& title, const char8_t* icon);
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
    // Icon accessors (font-glyph string or short label)
    const std::string& getIcon() const;
    void setIcon(const std::string& icon);
    void setIcon(const char8_t* icon);

    // Title as displayed in ImGui windows (may include icon/placeholder)
    std::string displayTitle() const;
    
protected:
    std::string title;
    std::string icon;
    bool isOpen;
};
