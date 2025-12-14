#pragma once

#include "Widget.hpp"
#include <glm/glm.hpp>
#include <imgui.h>

class SkyWidget : public Widget {
private:
    glm::vec3 horizonColor = glm::vec3(0.6f, 0.7f, 0.9f);
    glm::vec3 zenithColor = glm::vec3(0.05f, 0.15f, 0.4f);
    float warmth = 0.0f; // how warm the horizon becomes when sun is low
    float exponent = 1.0f; // gradient power
    float sunFlare = 1.0f; // sun flare intensity
    // Night parameters
    glm::vec3 nightHorizon = glm::vec3(0.02f, 0.03f, 0.06f);
    glm::vec3 nightZenith = glm::vec3(0.0f, 0.02f, 0.08f);
    float nightIntensity = 1.0f; // how dark/strong the night colors are
    float starIntensity = 0.5f; // brightness/amount of stars

public:
    SkyWidget() : Widget("Sky") {}

    void render() override {
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

    glm::vec3 getHorizonColor() const { return horizonColor; }
    glm::vec3 getZenithColor() const { return zenithColor; }
    float getWarmth() const { return warmth; }
    float getExponent() const { return exponent; }
    float getSunFlare() const { return sunFlare; }
    glm::vec3 getNightHorizon() const { return nightHorizon; }
    glm::vec3 getNightZenith() const { return nightZenith; }
    float getNightIntensity() const { return nightIntensity; }
    float getStarIntensity() const { return starIntensity; }
};
