#include "WaterWidget.hpp"
#include <imgui.h>

WaterWidget::WaterWidget(WaterRenderer* renderer, std::vector<WaterParams>* params)
    : Widget("Water Settings", u8"\uf043"), renderer(renderer), params(params) {
    isOpen = false;
}

void WaterWidget::render() {
    if (!isOpen || !renderer || !params) return;

    // Clamp current layer to valid range
    uint32_t count = static_cast<uint32_t>(params->size());
    if (count == 0) return;
    if (currentLayer < 0) currentLayer = 0;
    if ((uint32_t)currentLayer >= count) currentLayer = static_cast<int>(count - 1);

    WaterParams &layerParams = (*params)[currentLayer];

    if (ImGui::Begin(displayTitle().c_str(), &isOpen)) {
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
            ImGui::SliderFloat("Wave Scale", &layerParams.waveScale, 0.001f, 0.1f);
            ImGui::SliderFloat("Wave Height", &layerParams.bumpAmplitude, 0.0f, 256.0f);
            ImGui::SliderFloat("Wave Depth Transition", &layerParams.waveDepthTransition, 0.0f, 100.0f, "%.1f");
            ImGui::SetItemTooltip("Distance (world units) over which waves ramp from zero to full height.\n0 = disabled (no depth-based attenuation).");
            ImGui::SliderFloat("Noise Scale", &layerParams.noiseScale, 0.01f, 256.0f, "%.2f");
            ImGui::SliderInt("Noise Octaves", &layerParams.noiseOctaves, 1, 8);
            ImGui::SliderFloat("Noise Persistence", &layerParams.noisePersistence, 0.1f, 0.9f);
            ImGui::SliderFloat("Noise Time Speed", &layerParams.noiseTimeSpeed, 0.0f, 5.0f);
        }

        // Refraction settings
        if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Refraction", &layerParams.enableRefraction);
            ImGui::SetItemTooltip("Toggle Perlin noise-based refraction distortion on the underwater scene.");
            ImGui::SliderFloat("Refraction Strength", &layerParams.refractionStrength, 0.0f, 0.5f);
            ImGui::SliderFloat("Transparency", &layerParams.transparency, 0.0f, 1.0f);
            ImGui::SliderFloat("Water Tint", &layerParams.waterTint, 0.0f, 1.0f);
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
            ImGui::SetItemTooltip("How much environment reflection mixes into the surface.\n0 = no reflection, 1 = full mirror.");
            ImGui::SliderFloat("Fresnel Power", &layerParams.fresnelPower, 1.0f, 10.0f);
            ImGui::SetItemTooltip("Controls angle-dependence of reflection.\nHigher = reflection only at grazing angles.");
            ImGui::Checkbox("Uniform Reflection (no Fresnel)", &layerParams.uniformReflection);
            ImGui::SetItemTooltip("When enabled, reflection is applied uniformly by `Reflection Strength`\ninstead of being modulated by Fresnel.");
            ImGui::SliderFloat("Specular Intensity", &layerParams.specularIntensity, 0.0f, 10.0f);
            ImGui::SetItemTooltip("Brightness of the sun's specular highlight on the water.");
            ImGui::SliderFloat("Specular Power", &layerParams.specularPower, 8.0f, 512.0f, "%.0f");
            ImGui::SetItemTooltip("Sharpness of the specular highlight.\nHigher = tighter, smaller hotspot.");
            ImGui::SliderFloat("Glitter Intensity", &layerParams.glitterIntensity, 0.0f, 5.0f);
            ImGui::SetItemTooltip("Brightness of sun glitter sparkles on the water surface.");
        }

        // Blur settings
        if (ImGui::CollapsingHeader("Underwater Blur", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Blur", &layerParams.enableBlur);
            ImGui::SetItemTooltip("Apply a PCF box-blur to the refracted scene color\nfor a soft underwater look.");
            if (layerParams.enableBlur) {
                ImGui::SliderFloat("Blur Radius", &layerParams.blurRadius, 0.5f, 8.0f, "%.1f");
                ImGui::SetItemTooltip("Texel radius of the blur kernel.\nLarger values produce a wider blur.");
                ImGui::SliderInt("Blur Samples", &layerParams.blurSamples, 1, 8);
                ImGui::SetItemTooltip("Half-size of the NxN blur kernel.\nHigher values are smoother but more expensive.");
            }
        }

        // Water volume depth-based effects
        if (ImGui::CollapsingHeader("Water Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Blur and bump amplitude ramp up with water volume thickness "
                               "(back-face depth minus front-face depth).");
            ImGui::SliderFloat("Volume Blur Rate", &layerParams.volumeBlurRate, 0.0f, 1.0f, "%.3f");
            ImGui::SetItemTooltip("Exponential rate for blur increase with water thickness.\n"
                                  "0 = no depth-based blur modulation (full blur everywhere).\n"
                                  "Higher = blur reaches max faster with depth.");
            ImGui::SliderFloat("Volume Bump Rate", &layerParams.volumeBumpRate, 0.0f, 1.0f, "%.3f");
            ImGui::SetItemTooltip("Exponential rate for bump/wave amplitude increase with water thickness.\n"
                                  "0 = no depth-based bump modulation (full bump everywhere).\n"
                                  "Higher = bump reaches max faster with depth.");
            // volume light accumulation removed; caustics preserved below
        }

        // Caustics (light focusing) settings
        if (ImGui::CollapsingHeader("Caustics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Caustic Color", &layerParams.causticColor.x);
            ImGui::SliderFloat("Caustic Intensity", &layerParams.causticIntensity, 0.0f, 10.0f);
            ImGui::SetItemTooltip("Overall brightness multiplier for caustic highlights.");
            ImGui::SliderFloat("Caustic Scale", &layerParams.causticScale, 0.01f, 200.0f, "%.2f");
            ImGui::SetItemTooltip("Scale factor applied to the computed Jacobian determinant when forming caustic brightness.");
            ImGui::SliderFloat("Caustic Power", &layerParams.causticPower, 0.1f, 4.0f, "%.2f");
            ImGui::SetItemTooltip("Exponent to sharpen or soften caustic contrast.");
            ImGui::SliderFloat("Caustic Depth Scale", &layerParams.causticDepthScale, 0.1f, 256.0f, "%.2f");
            ImGui::SetItemTooltip("World-space depth (units) over which caustic blending transitions\nfrom surface-focused to bottom-focused. Lower values make bottom-based caustics appear\nfor shallower water.");
            ImGui::SliderFloat("Caustic Line Scale", &layerParams.causticLineScale, 0.01f, 10.0f, "%.2f");
            ImGui::SetItemTooltip("Multiplier applied to anisotropy to form thin caustic lines.\nHigher = thinner/more pronounced lines.");
            ImGui::SliderFloat("Caustic Line Mix", &layerParams.causticLineMix, 0.0f, 1.0f, "%.2f");
            ImGui::SetItemTooltip("Blend between cloudy caustics (0.0) and line-shaped caustics (1.0).");
            ImGui::SliderFloat("Caustic Velocity", &layerParams.causticVelocity, 0.0f, 1.0f, "%.3f");
            ImGui::SetItemTooltip("Speed multiplier for caustic animation. 0 = static.");
            // Caustic generation type (0 = Perlin, 1 = Voronoi)
            const char* causticTypes[] = { "Perlin (Jacobian)", "Voronoi (Worley)" };
            int ct = layerParams.causticType;
            if (ImGui::Combo("Caustic Type", &ct, causticTypes, IM_ARRAYSIZE(causticTypes))) {
                layerParams.causticType = ct;
            }
            ImGui::SetItemTooltip("Choose caustic generation: Perlin (jacobian-based) or Voronoi (Worley).");
        }

        ImGui::Separator();
        // Push updated params to GPU for the current layer
        renderer->updateGPUParamsForLayer(static_cast<uint32_t>(currentLayer), layerParams);
    }
    ImGui::End();
}
