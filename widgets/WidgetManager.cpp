#include "WidgetManager.hpp"

void WidgetManager::addWidget(std::shared_ptr<Widget> widget) {
	widgets.push_back(widget);
}

void WidgetManager::renderAll() {
	for (auto& widget : widgets) {
		if (widget->isVisible()) {
			widget->render();
		}
	}
}

void WidgetManager::renderMenu() {
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

std::shared_ptr<Widget> WidgetManager::getWidget(const std::string& title) {
	for (auto& widget : widgets) {
		if (widget->getTitle() == title) {
			return widget;
		}
	}
	return nullptr;
}
