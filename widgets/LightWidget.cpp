#include "LightWidget.hpp"

LightWidget::LightWidget(glm::vec3* lightDir)
	: Widget("Light Control"), lightDirection(lightDir) {
	if (lightDirection) {
		calculateAnglesFromDirection();
	}
}

void LightWidget::updateLightDirection() {
	if (!lightDirection) return;

	float azimuthRad = glm::radians(azimuth);
	float elevationRad = glm::radians(elevation);

	glm::vec3 direction;
	float cosE = cos(elevationRad);
	direction.x = cosE * sin(azimuthRad);
	direction.y = sin(elevationRad);
	direction.z = cosE * cos(azimuthRad);

	*lightDirection = glm::normalize(direction);
	std::cout << "[LightWidget] Az=" << azimuth << " El=" << elevation
			  << " Dir=(" << (*lightDirection).x << ", " << (*lightDirection).y << ", " << (*lightDirection).z << ")\n";
}

void LightWidget::calculateAnglesFromDirection() {
	if (!lightDirection) return;
	glm::vec3 dir = glm::normalize(*lightDirection);
	elevation = glm::degrees(asin(glm::clamp(dir.y, -1.0f, 1.0f)));
	azimuth = glm::degrees(atan2(dir.x, dir.z));
}

void LightWidget::render() {
	if (!isOpen) return;

	if (ImGui::Begin(title.c_str(), &isOpen)) {
		float dir[3] = { lightDirection->x, lightDirection->y, lightDirection->z };
		if (ImGui::InputFloat3("Direction", dir, "%.3f")) {
			glm::vec3 newDir = glm::normalize(glm::vec3(dir[0], dir[1], dir[2]));
			*lightDirection = newDir;
			calculateAnglesFromDirection();
		}
		ImGui::Text("Normalized: (%.3f, %.3f, %.3f)", lightDirection->x, lightDirection->y, lightDirection->z);
		ImGui::Text("Azimuth: %.1f°, Elevation: %.1f°", azimuth, elevation);

		ImGui::Separator();

		if (ImGui::SliderFloat("Azimuth", &azimuth, -180.0f, 180.0f, "%.1f°")) {
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Horizontal angle\n0° = North (+Z), 90° = East (+X), ±180° = South (-Z), -90° = West (-X)");
		}

		if (ImGui::SliderFloat("Elevation", &elevation, -89.0f, 89.0f, "%.1f°")) {
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Vertical angle\n0° = Horizon, +90° = Zenith (straight up), -90° = Nadir (straight down)");
		}

		ImGui::Separator();
		ImGui::Text("Presets:");

		if (ImGui::Button("Top-Down")) {
			azimuth = 0.0f;
			elevation = 89.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from directly above (zenith)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Diagonal")) {
			azimuth = 45.0f;
			elevation = 45.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("45° elevation from northeast");
		}

		ImGui::SameLine();
		if (ImGui::Button("Side")) {
			azimuth = 90.0f;
			elevation = 0.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the east (side, horizon)");
		}

		if (ImGui::Button("Front")) {
			azimuth = 0.0f;
			elevation = 0.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the north (front, horizon)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Back")) {
			azimuth = 180.0f;
			elevation = 0.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Light from the south (back, horizon)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Low Angle")) {
			azimuth = 45.0f;
			elevation = 15.0f;
			updateLightDirection();
		}
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("Low angle light for long shadows");
		}
	}
	ImGui::End();
}

float LightWidget::getAzimuth() const { return azimuth; }
float LightWidget::getElevation() const { return elevation; }

void LightWidget::setAzimuth(float a) { azimuth = a; updateLightDirection(); }
void LightWidget::setElevation(float e) { elevation = e; updateLightDirection(); }
