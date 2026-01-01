#include "LightWidget.hpp"

LightWidget::LightWidget(Light* l)
	: Widget("Light Control"), light(l) {
	if (light) {
		calculateAnglesFromDirection();
		glm::vec3 col = light->getColor();
		color[0] = col.r;
		color[1] = col.g;
		color[2] = col.b;
		intensity = light->getIntensity();
	}
}

void LightWidget::updateLight() {
	if (!light) return;

	light->setFromSpherical(azimuth, elevation);
	std::cout << "[LightWidget] Az=" << azimuth << " El=" << elevation
			  << " Dir=(" << light->getDirection().x << ", " << light->getDirection().y << ", " << light->getDirection().z << ")\n";
}

void LightWidget::calculateAnglesFromDirection() {
	if (!light) return;
	light->getSpherical(azimuth, elevation);
}

void LightWidget::render() {
	if (!isOpen) return;

	if (ImGui::Begin(title.c_str(), &isOpen)) {
		glm::vec3 dir = light->getDirection();
		float dirArray[3] = { dir.x, dir.y, dir.z };
		if (ImGui::InputFloat3("Direction", dirArray, "%.3f")) {
			glm::vec3 newDir = glm::normalize(glm::vec3(dirArray[0], dirArray[1], dirArray[2]));
			light->setDirection(newDir);
			calculateAnglesFromDirection();
		}
		ImGui::Text("Normalized: (%.3f, %.3f, %.3f)", dir.x, dir.y, dir.z);
		ImGui::Text("Azimuth: %.1f°, Elevation: %.1f°", azimuth, elevation);

		ImGui::Separator();

		if (ImGui::SliderFloat("Azimuth", &azimuth, -180.0f, 180.0f, "%.1f°")) {
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Horizontal angle\n0° = North (+Z), 90° = East (+X), ±180° = South (-Z), -90° = West (-X)");
		}

		if (ImGui::SliderFloat("Elevation", &elevation, -89.0f, 89.0f, "%.1f°")) {
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Vertical angle\n0° = Horizon, +90° = Zenith (straight up), -90° = Nadir (straight down)");
		}

		ImGui::Separator();
		ImGui::Text("Light Properties:");

		if (ImGui::ColorEdit3("Color", color)) {
			light->setColor(glm::vec3(color[0], color[1], color[2]));
		}

		if (ImGui::SliderFloat("Intensity", &intensity, 0.0f, 5.0f, "%.2f")) {
			light->setIntensity(intensity);
		}

		ImGui::Separator();
		ImGui::Text("Presets:");

		if (ImGui::Button("Top-Down")) {
			azimuth = 0.0f;
			elevation = 89.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from directly above (zenith)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Diagonal")) {
			azimuth = 45.0f;
			elevation = 45.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("45° elevation from northeast");
		}

		ImGui::SameLine();
		if (ImGui::Button("Side")) {
			azimuth = 90.0f;
			elevation = 0.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the east (side, horizon)");
		}

		if (ImGui::Button("Front")) {
			azimuth = 0.0f;
			elevation = 0.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the north (front, horizon)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Back")) {
			azimuth = 180.0f;
			elevation = 0.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the south (back, horizon)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Low Angle")) {
			azimuth = 45.0f;
			elevation = 15.0f;
			updateLight();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Low angle light for long shadows");
		}
	}
	ImGui::End();
}

float LightWidget::getAzimuth() const { return azimuth; }
float LightWidget::getElevation() const { return elevation; }

void LightWidget::setAzimuth(float a) { azimuth = a; updateLight(); }
void LightWidget::setElevation(float e) { elevation = e; updateLight(); }
