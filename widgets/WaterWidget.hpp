#pragma once

#include "Widget.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include <imgui.h>

class WaterWidget : public Widget {
public:
    WaterWidget(WaterRenderer* renderer) : Widget("Water Settings"), renderer(renderer) {
        isOpen = false;
    }

    void render() override {
        if (!isOpen || !renderer) return;

        WaterParams &params = renderer->getParams();

        if (ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::Text("Water Rendering Parameters");
            ImGui::Separator();

            // Wave settings
            if (ImGui::CollapsingHeader("Wave Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Wave Speed", &params.waveSpeed, 0.0f, 2.0f);
                ImGui::SliderFloat("Wave Scale", &params.waveScale, 0.001f, 0.1f);
                ImGui::SliderFloat("Wave Height", &params.bumpAmplitude, 0.0f, 256.0f);
                ImGui::SliderFloat("Wave Depth Transition", &params.waveDepthTransition, 0.0f, 100.0f, "%.1f");
                ImGui::SetItemTooltip("Distance (world units) over which waves ramp from zero to full height.\n0 = disabled (no depth-based attenuation).");
                ImGui::SliderFloat("Noise Scale", &params.noiseScale, 1.0f, 256.0f);
                ImGui::SliderInt("Noise Octaves", &params.noiseOctaves, 1, 8);
                ImGui::SliderFloat("Noise Persistence", &params.noisePersistence, 0.1f, 0.9f);
                ImGui::SliderFloat("Noise Time Speed", &params.noiseTimeSpeed, 0.0f, 5.0f);
            }

            // Refraction settings
            if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Refraction", &params.enableRefraction);
                ImGui::SetItemTooltip("Toggle Perlin noise-based refraction distortion on the underwater scene.");
                ImGui::SliderFloat("Refraction Strength", &params.refractionStrength, 0.0f, 0.5f);
                ImGui::SliderFloat("Transparency", &params.transparency, 0.0f, 1.0f);
                ImGui::SliderFloat("Water Tint", &params.waterTint, 0.0f, 1.0f);
            }

            // Color settings
            if (ImGui::CollapsingHeader("Water Color", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::ColorEdit3("Shallow Color", &params.shallowColor.x);
                ImGui::ColorEdit3("Deep Color", &params.deepColor.x);
                ImGui::SliderFloat("Depth Falloff", &params.depthFalloff, 0.001f, 1.0f);
            }

            // Reflection settings
            if (ImGui::CollapsingHeader("Surface Reflection", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Reflection", &params.enableReflection);
                ImGui::SetItemTooltip("Toggle sky/environment reflection on the water surface.");
                ImGui::SliderFloat("Reflection Strength", &params.reflectionStrength, 0.0f, 1.0f);
                ImGui::SetItemTooltip("How much environment reflection mixes into the surface.\n0 = no reflection, 1 = full mirror.");
                ImGui::SliderFloat("Fresnel Power", &params.fresnelPower, 1.0f, 10.0f);
                ImGui::SetItemTooltip("Controls angle-dependence of reflection.\nHigher = reflection only at grazing angles.");
                ImGui::SliderFloat("Specular Intensity", &params.specularIntensity, 0.0f, 10.0f);
                ImGui::SetItemTooltip("Brightness of the sun's specular highlight on the water.");
                ImGui::SliderFloat("Specular Power", &params.specularPower, 8.0f, 512.0f, "%.0f");
                ImGui::SetItemTooltip("Sharpness of the specular highlight.\nHigher = tighter, smaller hotspot.");
                ImGui::SliderFloat("Glitter Intensity", &params.glitterIntensity, 0.0f, 5.0f);
                ImGui::SetItemTooltip("Brightness of sun glitter sparkles on the water surface.");
            }

            // Blur settings
            if (ImGui::CollapsingHeader("Underwater Blur", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Enable Blur", &params.enableBlur);
                ImGui::SetItemTooltip("Apply a PCF box-blur to the refracted scene color\nfor a soft underwater look.");
                if (params.enableBlur) {
                    ImGui::SliderFloat("Blur Radius", &params.blurRadius, 0.5f, 8.0f, "%.1f");
                    ImGui::SetItemTooltip("Texel radius of the blur kernel.\nLarger values produce a wider blur.");
                    ImGui::SliderInt("Blur Samples", &params.blurSamples, 1, 8);
                    ImGui::SetItemTooltip("Half-size of the NxN blur kernel.\nHigher values are smoother but more expensive.");
                }
            }

            // Water volume depth-based effects
            if (ImGui::CollapsingHeader("Water Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextWrapped("Blur and bump amplitude ramp up with water volume thickness "
                                   "(back-face depth minus front-face depth).");
                ImGui::SliderFloat("Volume Blur Rate", &params.volumeBlurRate, 0.0f, 1.0f, "%.3f");
                ImGui::SetItemTooltip("Exponential rate for blur increase with water thickness.\n"
                                      "0 = no depth-based blur modulation (full blur everywhere).\n"
                                      "Higher = blur reaches max faster with depth.");
                ImGui::SliderFloat("Volume Bump Rate", &params.volumeBumpRate, 0.0f, 1.0f, "%.3f");
                ImGui::SetItemTooltip("Exponential rate for bump/wave amplitude increase with water thickness.\n"
                                      "0 = no depth-based bump modulation (full bump everywhere).\n"
                                      "Higher = bump reaches max faster with depth.");
            }

            ImGui::Separator();
            ImGui::Text("Time: %.2f", params.time);
        }
        ImGui::End();
    }

private:
    WaterRenderer* renderer;
};
