#pragma once

#include "Widget.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include <imgui.h>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterParams* params) : Widget("Water Settings"), params(params) {
        isOpen = false;
    }

    void render() override {
        if (!isOpen || !params) return;

        if (ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::Text("Water Rendering Parameters");
            ImGui::Separator();

            // Wave settings
            if (ImGui::CollapsingHeader("Wave Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Wave Speed", &params->waveSpeed, 0.0f, 2.0f);
                ImGui::SliderFloat("Wave Scale", &params->waveScale, 0.001f, 0.1f);
                ImGui::SliderFloat("Noise Scale", &params->noiseScale, 1.0f, 32.0f);
                ImGui::SliderInt("Noise Octaves", &params->noiseOctaves, 1, 8);
                ImGui::SliderFloat("Noise Persistence", &params->noisePersistence, 0.1f, 0.9f);
            }

            // Refraction settings
            if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Refraction Strength", &params->refractionStrength, 0.0f, 0.5f);
                ImGui::SliderFloat("Transparency", &params->transparency, 0.0f, 1.0f);
            }

            // Color settings
            if (ImGui::CollapsingHeader("Water Color", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::ColorEdit3("Shallow Color", &params->shallowColor.x);
                ImGui::ColorEdit3("Deep Color", &params->deepColor.x);
                ImGui::SliderFloat("Depth Falloff", &params->depthFalloff, 0.001f, 1.0f);
            }

            // Fresnel settings
            if (ImGui::CollapsingHeader("Surface Reflection")) {
                ImGui::SliderFloat("Fresnel Power", &params->fresnelPower, 1.0f, 10.0f);
            }

            ImGui::Separator();
            ImGui::Text("Time: %.2f", params->time);
        }
        ImGui::End();
    }

private:
    WaterParams* params;
};
