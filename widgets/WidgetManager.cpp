#include "WidgetManager.hpp"
#include <algorithm>

void WidgetManager::addWidget(std::shared_ptr<Widget> widget) {
	widgets.push_back(widget);
	std::sort(widgets.begin(), widgets.end(), [](const std::shared_ptr<Widget>& a, const std::shared_ptr<Widget>& b) {
		return a->getTitle() < b->getTitle();
	});
}

void WidgetManager::renderAll() {
	for (auto& widget : widgets) {
		if (!widget) continue;
		if (widget->isVisible()) {
			widget->render();
		}
	}
}

void WidgetManager::renderMenu() {
	if (ImGui::BeginMenu("Windows")) {
		for (auto& widget : widgets) {
			if (!widget) continue;
			bool isOpen = widget->isVisible();
			std::string label = widget->displayTitle();
			if (ImGui::MenuItem(label.c_str(), nullptr, &isOpen)) {
				if (isOpen) widget->show();
				else widget->hide();
			}
		}
		ImGui::EndMenu();
	}
}

std::shared_ptr<Widget> WidgetManager::getWidget(const std::string& title) {
	for (auto& widget : widgets) {
		if (!widget) continue;
		if (widget->getTitle() == title) {
			return widget;
		}
	}
	return nullptr;
}
