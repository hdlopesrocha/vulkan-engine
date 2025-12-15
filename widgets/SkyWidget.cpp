#include "SkyWidget.hpp"

SkyWidget::SkyWidget() : Widget("Sky") {}

void SkyWidget::render() {
	if (!isOpen) return;
	if (ImGui::Begin(title.c_str(), &isOpen)) {
		float hc[3] = { horizonColor.r, horizonColor.g, horizonColor.b };
		if (ImGui::ColorEdit3("Horizon Color", hc)) {
			horizonColor = glm::vec3(hc[0], hc[1], hc[2]);
		}
		float zc[3] = { zenithColor.r, zenithColor.g, zenithColor.b };
		if (ImGui::ColorEdit3("Zenith Color", zc)) {
			zenithColor = glm::vec3(zc[0], zc[1], zc[2]);
		}
		ImGui::Separator();
		ImGui::Text("Night settings:");
		float nh[3] = { nightHorizon.r, nightHorizon.g, nightHorizon.b };
		if (ImGui::ColorEdit3("Night Horizon", nh)) nightHorizon = glm::vec3(nh[0], nh[1], nh[2]);
		float nz[3] = { nightZenith.r, nightZenith.g, nightZenith.b };
		if (ImGui::ColorEdit3("Night Zenith", nz)) nightZenith = glm::vec3(nz[0], nz[1], nz[2]);
		ImGui::SliderFloat("Night Intensity", &nightIntensity, 0.0f, 1.0f);
		ImGui::SliderFloat("Star Intensity", &starIntensity, 0.0f, 2.0f);
		ImGui::Separator();
		ImGui::SliderFloat("Warmth", &warmth, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat("Exponent", &exponent, 0.1f, 5.0f, "%.2f");
		ImGui::SliderFloat("Sun Flare", &sunFlare, 0.0f, 2.0f, "%.2f");
		ImGui::Text("Note: Light direction is controlled by Light widget");
	}
	ImGui::End();
}

glm::vec3 SkyWidget::getHorizonColor() const { return horizonColor; }
glm::vec3 SkyWidget::getZenithColor() const { return zenithColor; }
float SkyWidget::getWarmth() const { return warmth; }
float SkyWidget::getExponent() const { return exponent; }
float SkyWidget::getSunFlare() const { return sunFlare; }
glm::vec3 SkyWidget::getNightHorizon() const { return nightHorizon; }
glm::vec3 SkyWidget::getNightZenith() const { return nightZenith; }
float SkyWidget::getNightIntensity() const { return nightIntensity; }
float SkyWidget::getStarIntensity() const { return starIntensity; }
