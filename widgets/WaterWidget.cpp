#include "WaterWidget.hpp"
#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

WaterWidget::WaterWidget(WaterRenderer* renderer_, std::vector<WaterParams>* params_)
    : Widget("Water Settings", u8"\uf043"), renderer(renderer_), params(params_) {
    isOpen = false;
}

void WaterWidget::render() {
    if (!renderer || !params) return;

    // Clamp current layer to valid range
    uint32_t count = static_cast<uint32_t>(params->size());
    if (count == 0) return;
    if (currentLayer < 0) currentLayer = 0;
    if ((uint32_t)currentLayer >= count) currentLayer = static_cast<int>(count - 1);

    WaterParams &layerParams = (*params)[currentLayer];

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;
        ImGui::Text("Water Rendering Parameters");
        // Pagination controls for multiple water param layers
        if (count > 1) {
            ImGui::Text("Layer: %d / %d", currentLayer, count - 1);
            ImGui::SameLine();
            if (ImGui::Button("Prev") && currentLayer > 0) { currentLayer--; }
            ImGui::SameLine();
            if (ImGui::Button("Next") && currentLayer + 1 < (int)count) { currentLayer++; }
            ImGui::SameLine();
            ImGui::SliderInt("Layer Index", &currentLayer, 0, static_cast<int>(count - 1));
            ImGui::Separator();
        }
        ImGui::Separator();

        // Wave settings
        if (ImGui::CollapsingHeader("Wave Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Wave Speed", &layerParams.waveSpeed, 0.0f, 2.0f);
            ImGui::SliderFloat("Wave Height", &layerParams.bumpAmplitude, 0.0f, 256.0f);
            ImGui::SliderFloat("Wave Depth Transition", &layerParams.waveDepthTransition, 0.0f, 100.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Distance (world units) over which waves ramp from zero to full height.\n0 = disabled (no depth-based attenuation).");
            ImGui::SliderFloat("Noise Scale", &layerParams.noiseScale, 0.01f, 256.0f, "%.2f");
            ImGui::SliderInt("Noise Octaves", &layerParams.noiseOctaves, 1, 8);
            ImGui::SliderFloat("Noise Persistence", &layerParams.noisePersistence, 0.1f, 0.9f);
            ImGui::SliderFloat("Noise Lacunarity", &layerParams.noiseLacunarity, 1.0f, 4.0f);
            ImGui::SliderFloat("Noise Time Speed", &layerParams.noiseTimeSpeed, 0.0f, 5.0f);
        }

        // Tessellation settings
        if (ImGui::CollapsingHeader("Tessellation", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Near Distance", &layerParams.tessNearDist, 1.0f, 500.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Camera distance at which tessellation reaches maximum level.");
            ImGui::SliderFloat("Far Distance", &layerParams.tessFarDist, 1.0f, 2000.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Camera distance at which tessellation drops to minimum level.");
            ImGui::SliderFloat("Min Level", &layerParams.tessMinLevel, 1.0f, 32.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Minimum tessellation factor (far away / flat areas).");
            ImGui::SliderFloat("Max Level", &layerParams.tessMaxLevel, 1.0f, 64.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Maximum tessellation factor (close up / active wave areas).");
            ImGui::SliderFloat("Noise Influence", &layerParams.tessNoiseInfluence, 0.0f, 1.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("How much the wave noise pattern affects tessellation.\n0 = uniform distance-based, 1 = fully noise-adaptive.");
        }

        // Refraction settings
        if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Refraction", &layerParams.enableRefraction);
            ImGuiHelpers::SetTooltipIfHovered("Toggle Perlin noise-based refraction distortion on the underwater scene.");
            ImGui::SliderFloat("Refraction Strength", &layerParams.refractionStrength, 0.0f, 0.5f);
            ImGui::SliderFloat("Transparency", &layerParams.transparency, 0.0f, 1.0f);
            ImGui::SliderFloat("Water Tint", &layerParams.waterTint, 0.0f, 1.0f);
            ImGui::SliderFloat("Water IOR", &layerParams.ior, 1.0f, 1.6f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Index of refraction for Snell air<->water bending (physical water = 1.333).");
            ImGui::SliderFloat3("Absorption (RGB)", &layerParams.absorption.x, 0.0f, 2.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Beer-Lambert absorption coefficients: how fast refracted light fades with water depth.");
            ImGui::SliderFloat("Absorption Scale", &layerParams.absorptionScale, 0.0f, 4.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Thickness multiplier for absorption (tuning).");
            ImGui::SliderFloat("Max Thickness", &layerParams.maxThickness, 0.5f, 20.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Clamp for RT hit thickness: kills far-hit blackouts, keeps deep ground visible.");
            ImGui::SliderFloat("Shore Fade Depth", &layerParams.shoreFadeDepth, 0.0f, 2.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Water depth over which the shoreline fades from fully transparent (waterline shows the bottom with no water color).");
        }

        // Color settings
        if (ImGui::CollapsingHeader("Water Color", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Shallow Color", &layerParams.shallowColor.x);
            ImGui::ColorEdit3("Deep Color", &layerParams.deepColor.x);
            ImGui::SliderFloat("Depth Falloff", &layerParams.depthFalloff, 0.001f, 1.0f);
        }

        // Reflection settings
        if (ImGui::CollapsingHeader("Surface Reflection", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Reflection", &layerParams.enableReflection);
            ImGui::SetItemTooltip("Toggle sky/environment reflection on the water surface.");
            ImGui::SliderFloat("Reflection Strength", &layerParams.reflectionStrength, 0.0f, 1.0f);
            ImGuiHelpers::SetTooltipIfHovered("How much environment reflection mixes into the surface.\n0 = no reflection, 1 = full mirror.");
            ImGui::SliderFloat("Fresnel Power", &layerParams.fresnelPower, 1.0f, 10.0f);
            ImGuiHelpers::SetTooltipIfHovered("Controls angle-dependence of reflection.\nHigher = reflection only at grazing angles.");
            ImGui::Checkbox("Uniform Reflection (no Fresnel)", &layerParams.uniformReflection);
            ImGuiHelpers::SetTooltipIfHovered("When enabled, reflection is applied uniformly by `Reflection Strength`\ninstead of being modulated by Fresnel.");
            ImGui::SliderFloat("Specular Intensity", &layerParams.specularIntensity, 0.0f, 10.0f);
            ImGuiHelpers::SetTooltipIfHovered("Brightness of the sun's specular highlight on the water.");
            ImGui::SliderFloat("Specular Power", &layerParams.specularPower, 8.0f, 512.0f, "%.0f");
            ImGui::SetItemTooltip("Sharpness of the specular highlight.\nHigher = tighter, smaller hotspot.");
            ImGui::SliderFloat("Glitter Intensity", &layerParams.glitterIntensity, 0.0f, 5.0f);
            ImGui::SetItemTooltip("Brightness of sun glitter sparkles on the water surface.");
        }

        // Water volume depth-based effects
        if (ImGui::CollapsingHeader("Water Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Bump amplitude ramps up with water volume thickness "
                               "(back-face depth minus front-face depth).");
            ImGui::SliderFloat("Volume Bump Rate", &layerParams.volumeBumpRate, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Exponential rate for bump/wave amplitude increase with water thickness.\n"
                                  "0 = no depth-based bump modulation (full bump everywhere).\n"
                                  "Higher = bump reaches max faster with depth.");
            // volume light accumulation removed; caustics preserved below
        }

        // Caustics (physical sunlight focusing; wave-shape driven)
        if (ImGui::CollapsingHeader("Caustics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Physically derived from the wave height field: sunlight refracted by the "
                               "surface focuses on the bottom with irradiance E/E0 = 1/|1 + d*K*d2h/du2| "
                               "(Snell K, water column d, wave curvature along the sun azimuth u). "
                               "The pattern, its scale and its motion ride the waves — no separate "
                               "caustic noise or speed.");
            ImGui::ColorEdit3("Caustic Color", &layerParams.causticColor.x);
            ImGuiHelpers::SetTooltipIfHovered("Tint of the focused sunlight (near-white is physical; "
                                              "colored tints are stylistic).");
            ImGui::SliderFloat("Caustic Intensity", &layerParams.causticIntensity, 0.0f, 4.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Strength of the added focused light (excess over the flat-surface "
                                              "irradiance, attenuated by the water column). 0 disables caustics.");
            ImGui::SliderFloat("Caustic Softness", &layerParams.causticSoftness, 0.02f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Clamp floor on |J| (the inverse-Jacobian fold is unbounded).\n"
                                              "Lower = sharper, brighter caustic ridges; 1.0 disables them.");
            ImGui::SliderFloat("Caustic Depth Scale", &layerParams.causticDepthScale, 0.1f, 256.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Depth reference for the water-tint volume ramp (shallow->deep blend).");
        }

        ImGui::Separator();
        // Push updated params to GPU for the current layer.
        renderer->updateGPUParamsForLayer(static_cast<uint32_t>(currentLayer), layerParams);
    }
