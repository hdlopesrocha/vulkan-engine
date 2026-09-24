#include "WaterWidget.hpp"
#include <imgui.h>
#include <algorithm>
#include <cfloat>
#include "components/ColumnLayout.hpp"
#include "components/ImGuiHelpers.hpp"

WaterWidget::WaterWidget(WaterRenderer* renderer_, std::vector<WaterParams>* params_)
    : Widget("Water Settings", u8"\uf043"), renderer(renderer_), params(params_) {
    isOpen = false;
}

void WaterWidget::render() {
    using namespace ImGuiComponents;

    if (!renderer || !params) return;

    uint32_t count = static_cast<uint32_t>(params->size());
    if (count == 0) return;
    if (currentLayer < 0) currentLayer = 0;
    if ((uint32_t)currentLayer >= count) currentLayer = static_cast<int>(count - 1);

    WaterParams &layerParams = (*params)[currentLayer];

    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1280, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    std::vector<ColumnSection> sections;
    sections.reserve(23);

    sections.push_back({[&]() {
        ImGui::Text("Water Layer");
        ColSeparator();
        if (count > 1) {
            ImGui::Text("Layer %d / %d", currentLayer, static_cast<int>(count - 1));
            if (ImGui::Button("Prev") && currentLayer > 0) { currentLayer--; }
            ImGui::SameLine();
            if (ImGui::Button("Next") && currentLayer + 1 < static_cast<int>(count)) { currentLayer++; }
            FieldLabel("Layer Index", "Water layer to edit (0-based).");
            ImGui::SetNextItemWidth(kColumnWidth);
            ImGui::PushID("Layer Index");
            ImGui::SliderInt("##v", &currentLayer, 0, static_cast<int>(count - 1));
            ImGui::PopID();
        } else {
            ImGui::TextWrapped("Single water layer (index 0).");
        }
    }});

    // ── Water depth regions: one subsection per region, ordered deep → shore.
    // Each subsection carries the region's tint color and the distance/depth
    // that defines it. The boundaries are the same thickness zones the wave
    // system uses, so they are the single source of truth for both. ──
    sections.push_back({[&]() {
        ImGui::Text("Water");
        ColSeparator();
        ImGui::TextWrapped("One water region: a single tint colour, and one sine swell that travels "
                           "toward the shore and fades out before the waterline.");
        ColorEdit3Field("Water Color", &layerParams.waterColor.x,
            "The water tint (one region - there is no depth-band palette any more).");
        SliderFloatField("Tint Shore Fade (m)", &layerParams.tintShoreFadeDepth, 0.0f, 8.0f, "%.2f",
            "Water depth over which the tint fades to 0 at the waterline.");
        SliderFloatField("Shore Wave Fade (m)", &layerParams.shoreWaveFade, 0.0f, 512.0f, "%.1f",
            "Water depth over which the swell fades out approaching the shore.\n"
            "0 = no fade (the sine runs to the waterline).");
        SliderFloatField("Wave Steepness", &layerParams.waveSteepness, 0.0f, 0.95f, "%.2f",
            "Gerstner pinch: how sharp and forward-leaning the crests are.\n"
            "0 = a pure sine (symmetric crests).");
        SliderFloatField("Crest Sharpness", &layerParams.waveCrestSharpness, 1.0f, 4.0f, "%.2f",
            "Exponent on the wave profile: 1 = plain Gerstner, higher = narrower,\n"
            "spikier crests and troughs with steeper faces between them.");
        SliderFloatField("Shore Slope (m/m)", &layerParams.shoreWaveSlope, 0.002f, 0.2f, "%.4f",
            "Beach slope: metres of depth per metre from the shore. The wave phase runs\n"
            "on depth / slope, i.e. on the distance to the waterline, so the crests follow\n"
            "the depth contours (the shoreline).");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Shore Waves");
        ColSeparator();
        CheckboxField("Enable Waves", &layerParams.enableWaves,
            "Master toggle for the sine swell.\n"
            "Only the first water material enables it by default.");
        SliderFloatField("Wave Height", &layerParams.waveAmplitude, 0.0f, 64.0f, "%.2f",
            "Overall vertical amplitude of the wave displacement (world units).");
        SliderFloatField("Wave Speed (m/s)", &layerParams.waveSpeed, 0.0f, 30.0f, "%.2f",
            "Phase speed of the sine swell.");
        SliderFloatField("Wave Period", &layerParams.wavePeriod, 5.0f, 4096.0f, "%.1f",
            "Wavelength of the sine swell (world units).\n"
            "The swell's HEIGHT follows the wavelength (constant steepness), so this\n"
            "scales the whole swell up, not just its wavelength.",
            ImGuiSliderFlags_Logarithmic);
        SliderFloatField("Shore Direction (deg)", &layerParams.shoreWaveAngle, 0.0f, 360.0f, "%.1f",
            "Direction the sine wave travels, toward the shore.\n"
            "0 = +Z, 90 = +X, 180 = -Z, 270 = -X.");
        SliderFloatField("Shore Gradient Step", &layerParams.shoreGradientStep, 0.0f, 64.0f, "%.1f",
            "Screen-texel step used to measure the local direction of decreasing depth\n"
            "(the direction to the shore). The wave faces aim along it, so the swell\n"
            "runs head-on into every part of the coastline. 0 = use the fixed angle above.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Foam");
        ColSeparator();
        CheckboxField("Enable Foam", &layerParams.enableFoam);
        ColorEdit3Field("Foam Color", &layerParams.foamColor.x);
        SliderFloatField("Foam Amount", &layerParams.foamColorAmount, 0.0f, 1.0f, "%.2f");
        SliderFloatField("Crest Threshold", &layerParams.foamCrestThreshold, 0.0f, 0.99f, "%.3f",
            "Wave crest height at which foam starts appearing.");
        SliderFloatField("Trail Lag", &layerParams.foamTrailPhase, 0.0f, 2.0f, "%.3f",
            "Lag of the trailing foam band behind the ridged crest\n"
            "(ridge-feature units; 1 = one crest feature).");
        SliderFloatField("Foam Edge", &layerParams.foamEdge, 0.0f, 1.0f, "%.3f",
            "Hardness of the foam mask edges.\n"
            "0 = soft gradients, 1 = hard, well-defined foam edges.");
        SliderFloatField("Foam Coverage", &layerParams.foamCoverage, 0.0f, 1.0f, "%.3f",
            "Global foam coverage multiplier (lighter/airier foam < 1).");
        SliderFloatField("Foam Decay", &layerParams.foamDecay, 0.0f, 0.5f, "%.4f",
            "Foam extinction per meter below the breaker line.");
        SliderFloatField("Shore Foam (legacy)", &layerParams.foamShoreAmount, 0.0f, 1.0f, "%.3f",
            "LEGACY - no effect. The shoreline line is the Contact Foam below.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Contact Foam");
        ColSeparator();
        SliderFloatField("Contact Width", &layerParams.foamContactWidth, 0.05f, 32.0f, "%.2f",
            "Depth band of the final foam line where the water meets the\n"
            "solid (peaks at depth 0, world units).");
        SliderFloatField("Contact Amount", &layerParams.foamContactAmount, 0.0f, 1.0f, "%.3f");
        SliderFloatField("Contact Opacity", &layerParams.foamContactAlpha, 0.0f, 1.0f, "%.3f",
            "Minimum composite opacity forced for the contact line so the\n"
            "last water pixels still render foam.");
        SliderFloatField("Contact Pulse Floor", &layerParams.foamContactFloor, 0.0f, 1.0f, "%.3f",
            "Residual contact-foam strength between incoming crests.\n"
            "0 = the line fully retreats and arrives with each wave,\n"
            "1 = continuous line.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Foam Noise");
        ColSeparator();
        SliderFloatField("Foam Noise Period", &layerParams.foamNoisePeriod, 1.0f, 4096.0f, "%.2f",
            "Foam breakup feature period (world units).",
            ImGuiSliderFlags_Logarithmic);
        SliderFloatField("Foam Noise Speed", &layerParams.foamNoiseSpeed, 0.0f, 2.0f, "%.3f");
        SliderFloatField("Foam Noise Amount", &layerParams.foamNoiseAmount, 0.0f, 1.0f, "%.3f");
        SliderFloatField("Foam Mask Floor", &layerParams.foamMaskFloor, 0.0f, 1.0f, "%.3f");
        SliderFloatField("Foam Diffuse Floor", &layerParams.foamDiffuseFloor, 0.0f, 1.0f, "%.3f");
        SliderFloatField("Foam Ambient", &layerParams.foamAmbient, 0.0f, 1.0f, "%.3f");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Noise Detail");
        ColSeparator();
        SliderFloatField("Noise Period", &layerParams.noisePeriod, 0.25f, 4096.0f, "%.3f",
            "Chop/refraction/specular noise feature period (world units).\n"
            "Larger = broader AND taller chop (its height follows the wavelength,\n"
            "constant steepness), so this is a feature-SIZE knob. 0 = disabled.",
            ImGuiSliderFlags_Logarithmic);
        SliderIntField("Noise Octaves", &layerParams.noiseOctaves, 1, 8);
        SliderFloatField("Noise Persistence", &layerParams.noisePersistence, 0.1f, 0.9f);
        SliderFloatField("Noise Lacunarity", &layerParams.noiseLacunarity, 1.0f, 4.0f);
        SliderFloatField("Noise Time Speed", &layerParams.noiseTimeSpeed, 0.0f, 5.0f);
    }});

    sections.push_back({[&]() {
        ImGui::Text("Volumetric Scattering");
        ColSeparator();
        ImGui::TextWrapped("Single-scattering sunlight inside the measured water column "
                           "(Henyey-Greenstein phase, saturating with thickness).");
        CheckboxField("Enable Volumetric", &layerParams.enableVolumetric);
        ColorEdit3Field("Scatter Color", &layerParams.volumetricColor.x);
        SliderFloatField("Scatter Strength", &layerParams.volumetricStrength, 0.0f, 2.0f, "%.3f");
        SliderFloatField("Scatter Density", &layerParams.volumetricDensity, 0.0f, 2.0f, "%.4f");
        SliderFloatField("Scatter Anisotropy", &layerParams.volumetricPhaseG, -0.95f, 0.95f, "%.3f",
            "Henyey-Greenstein g: 0 = isotropic, >0 = forward scattering (sun halo).");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Tessellation");
        ColSeparator();
        SliderFloatField("Near Distance", &layerParams.tessNearDist, 1.0f, 500.0f, "%.1f",
            "Camera distance at which tessellation reaches maximum level.");
        SliderFloatField("Far Distance", &layerParams.tessFarDist, 1.0f, 2000.0f, "%.1f",
            "Camera distance at which tessellation drops to minimum level.");
        SliderFloatField("Min Level", &layerParams.tessMinLevel, 1.0f, 32.0f, "%.1f",
            "Minimum tessellation factor (far away / flat areas).");
        SliderFloatField("Max Level", &layerParams.tessMaxLevel, 1.0f, 64.0f, "%.1f",
            "Maximum tessellation factor (close up / active wave areas).");
        SliderFloatField("Noise Influence", &layerParams.tessNoiseInfluence, 0.0f, 1.0f, "%.2f",
            "How much the wave noise pattern affects tessellation.\n0 = uniform distance-based, 1 = fully noise-adaptive.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Refraction");
        ColSeparator();
        CheckboxField("Enable Refraction", &layerParams.enableRefraction,
            "Toggle Perlin noise-based refraction distortion on the underwater scene.\n"
            "Off = no distortion/Snell look: the water stays transparent and the\n"
            "solid bottom renders through it undistorted.");
        SliderFloatField("Refraction Strength", &layerParams.refractionStrength, 0.0f, 0.5f);
        SliderFloatField("Transparency", &layerParams.transparency, 0.0f, 1.0f);
        SliderFloatField("Water Tint", &layerParams.waterTint, 0.0f, 1.0f);
        SliderFloatField("Water IOR", &layerParams.ior, 1.0f, 1.6f, "%.3f",
            "Index of refraction for Snell air<->water bending (physical water = 1.333).");
        SliderFloat3Field("Absorption (RGB)", &layerParams.absorption.x, 0.0f, 2.0f, "%.3f",
            "Beer-Lambert absorption coefficients: how fast refracted light fades with water depth.");
        SliderFloatField("Absorption Scale", &layerParams.absorptionScale, 0.0f, 4.0f, "%.2f",
            "Thickness multiplier for absorption (tuning).");
        SliderFloatField("Max Thickness", &layerParams.maxThickness, 0.5f, 20.0f, "%.1f",
            "Clamp for RT hit thickness: kills far-hit blackouts, keeps deep ground visible.");
        SliderFloatField("Shore Fade Depth", &layerParams.shoreFadeDepth, 0.0f, 2.0f, "%.2f",
            "Water depth over which the shoreline fades from fully transparent (waterline shows the bottom with no water color).");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Refraction / Tint Blur");
        ColSeparator();
        ImGui::TextWrapped("Per-material blur of the refracted bottom + tint in the final pass. "
                           "The radius grows with the measured water depth up to the max, so "
                           "deep water softens while the shoreline stays crisp; reflections and "
                           "surface highlights are never blurred.");
        CheckboxField("Enable Blur", &layerParams.enableBlur,
            "Blur this water material's refraction + tint body in the final composite.\n"
            "Requires the global Settings 'Water Blur' toggle too.\n"
            "Reflections and surface highlights stay sharp.");
        SliderFloatField("Max Radius (px)", &layerParams.blurRadius, 0.0f, 32.0f, "%.1f",
            "Blur radius clamp for this material (pixels) at full depth.");
        SliderFloatField("Depth Rate (px/m)", &layerParams.blurDepthScale, 0.0f, 4.0f, "%.3f",
            "Blur radius growth per meter of measured water depth (pixels/m).");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Surface Reflection");
        ColSeparator();
        CheckboxField("Enable Reflection", &layerParams.enableReflection,
            "Toggle sky/environment reflection on the water surface.");
        SliderFloatField("Reflection Strength", &layerParams.reflectionStrength, 0.0f, 1.0f, "%.3f",
            "How much environment reflection mixes into the surface.\n0 = no reflection, 1 = full mirror.");
        SliderFloatField("Fresnel Power", &layerParams.fresnelPower, 1.0f, 10.0f, "%.3f",
            "Controls angle-dependence of reflection.\nHigher = reflection only at grazing angles.");
        CheckboxField("Uniform Reflection (no Fresnel)", &layerParams.uniformReflection,
            "When enabled, reflection is applied uniformly by `Reflection Strength`\ninstead of being modulated by Fresnel.");
        SliderFloatField("Specular Intensity", &layerParams.specularIntensity, 0.0f, 10.0f, "%.3f",
            "Brightness of the sun's specular highlight on the water.");
        SliderFloatField("Specular Power", &layerParams.specularPower, 8.0f, 512.0f, "%.0f",
            "Sharpness of the specular highlight.\nHigher = tighter, smaller hotspot.");
        SliderFloatField("Glitter Intensity", &layerParams.glitterIntensity, 0.0f, 5.0f, "%.3f",
            "Brightness of sun glitter sparkles on the water surface.");
    }});

    sections.push_back({[&]() {
        ImGui::Text("Caustics");
        ColSeparator();
        ImGui::TextWrapped("Physically derived from the wave height field: sunlight refracted by the "
                           "surface focuses on the bottom with irradiance E/E0 = 1/|1 + d*K*d2h/du2| "
                           "(Snell K, water column d, wave curvature along the sun azimuth u). "
                           "The pattern, its scale and its motion ride the waves — no separate "
                           "caustic noise or speed.");
        ColorEdit3Field("Caustic Color", &layerParams.causticColor.x,
            "Tint of the focused sunlight (near-white is physical; colored tints are stylistic).");
        SliderFloatField("Caustic Intensity", &layerParams.causticIntensity, 0.0f, 4.0f, "%.2f",
            "Strength of the added focused light (excess over the flat-surface "
            "irradiance, attenuated by the water column). 0 disables caustics.");
        SliderFloatField("Caustic Softness", &layerParams.causticSoftness, 0.02f, 1.0f, "%.3f",
            "Clamp floor on |J| (the inverse-Jacobian fold is unbounded).\n"
            "Lower = sharper, brighter caustic ridges; 1.0 disables them.");
    }});

    static std::vector<float> cachedH;
    const std::vector<float> estimates = {
        EstimateSectionHeight(3),   // 0  Water Layer
        EstimateSectionHeight(4),   // 1  Water Region Tint
        EstimateSectionHeight(3),   // 2  Region 1: Deep Ocean
        EstimateSectionHeight(3),   // 3  Region 2: Shoaling
        EstimateSectionHeight(3),   // 4  Region 3: Breaker Line
        EstimateSectionHeight(3),   // 5  Region 4: Foam Band
        EstimateSectionHeight(3),   // 6  Region 5: Shore Line
        EstimateSectionHeight(6),   // 7  Shore Waves
        EstimateSectionHeight(5),   // 8  Cross Swell
        EstimateSectionHeight(5),   // 9  Crest Detail
        EstimateSectionHeight(6),   // 10 Wave Shaping (by Zone)
        EstimateSectionHeight(6),   // 11 Breakers
        EstimateSectionHeight(5),   // 12 Organic Mask
        EstimateSectionHeight(11),  // 13 Foam
        EstimateSectionHeight(4),   // 14 Contact Foam
        EstimateSectionHeight(6),   // 15 Foam Noise
        EstimateSectionHeight(5),   // 16 Noise Detail
        EstimateSectionHeight(6),   // 17 Volumetric
        EstimateSectionHeight(5),   // 18 Tessellation
        EstimateSectionHeight(9),   // 19 Refraction
        EstimateSectionHeight(4),   // 20 Refraction / Tint Blur
        EstimateSectionHeight(7),   // 21 Surface Reflection
        EstimateSectionHeight(4),   // 22 Caustics
    };
    LayoutSections(sections, cachedH, estimates);

    renderer->updateGPUParamsForLayer(static_cast<uint32_t>(currentLayer), (*params)[currentLayer]);
}
