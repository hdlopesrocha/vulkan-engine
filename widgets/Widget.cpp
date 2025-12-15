#include "Widget.hpp"

Widget::Widget(const std::string& title) : title(title), isOpen(false) {}
Widget::~Widget() = default;

bool Widget::isVisible() const { return isOpen; }
void Widget::show() { isOpen = true; }
void Widget::hide() { isOpen = false; }
void Widget::toggle() { isOpen = !isOpen; }

const std::string& Widget::getTitle() const { return title; }
