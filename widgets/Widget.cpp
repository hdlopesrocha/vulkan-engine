#include "Widget.hpp"

Widget::Widget(const std::string& title_, const std::string& icon_)
	: title(title_), icon(icon_), isOpen(false) {}
// Accept char8_t (u8"...") literals by converting to char*
Widget::Widget(const std::string& title_, const char8_t* icon_)
    : title(title_), icon(reinterpret_cast<const char*>(icon_)), isOpen(false) {}
Widget::~Widget() = default;

bool Widget::isVisible() const { return isOpen; }
void Widget::show() { isOpen = true; }
void Widget::hide() { isOpen = false; }

const std::string& Widget::getTitle() const { return title; }

std::string Widget::displayTitle() const {
	if (!icon.empty()) return icon + std::string(" ") + title;
	// default placeholder when no icon is provided
	return std::string("? ") + title;
}
