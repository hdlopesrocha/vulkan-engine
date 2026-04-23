#include "Widget.hpp"

Widget::Widget(const std::string& title, const std::string& icon)
	: title(title), icon(icon), isOpen(false) {}
// Accept char8_t (u8"...") literals by converting to char*
Widget::Widget(const std::string& title, const char8_t* icon)
    : title(title), icon(reinterpret_cast<const char*>(icon)), isOpen(false) {}
Widget::~Widget() = default;

bool Widget::isVisible() const { return isOpen; }
void Widget::show() { isOpen = true; }
void Widget::hide() { isOpen = false; }
void Widget::toggle() { isOpen = !isOpen; }

const std::string& Widget::getTitle() const { return title; }

const std::string& Widget::getIcon() const { return icon; }
void Widget::setIcon(const std::string& i) { icon = i; }
void Widget::setIcon(const char8_t* i) { icon = reinterpret_cast<const char*>(i); }

std::string Widget::displayTitle() const {
	if (!icon.empty()) return icon + std::string(" ") + title;
	// default placeholder when no icon is provided
	return std::string("? ") + title;
}
