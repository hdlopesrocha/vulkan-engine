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

        // ── Shore waves (master + shape + direction) ──
        if (ImGui::CollapsingHeader("Shore Waves", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Waves", &layerParams.enableWaves);
            ImGuiHelpers::SetTooltipIfHovered("Master toggle for the thickness-zoned shore-wave system.\n"
                                              "Only the first water material enables it by default.");
            ImGui::SliderFloat("Wave Height", &layerParams.bumpAmplitude, 0.0f, 64.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Overall vertical amplitude of the wave displacement (world units).");
            ImGui::SliderFloat("Wave Speed", &layerParams.waveSpeed, 0.0f, 30.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Deep-water phase speed of the primary swell (m/s).");
            ImGui::SliderFloat("Wave Period", &layerParams.wavePeriod, 5.0f, 500.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Primary swell wavelength (world units). Larger = longer, slower waves.");
            ImGui::SliderFloat("Shore Direction (deg)", &layerParams.shoreWaveAngle, 0.0f, 360.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("FALLBACK wave propagation direction toward the shore.\n"
                                              "The shader normally derives the local shore direction from the\n"
                                              "water-depth gradient (toward thinning water); this angle is used\n"
                                              "only where the bottom cannot be measured.\n"
                                              "0 = +Z, 90 = +X, 180 = -Z, 270 = -X.");
            ImGui::SliderFloat("Shore Gradient Step", &layerParams.shoreGradientStep, 0.0f, 64.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Screen-texel step used to sample the water-depth gradient that\n"
                                              "yields the shore direction. Wider = more stable deep-ocean direction.\n"
                                              "0 = disable the gradient and always use the fixed Shore Direction angle.");
            ImGui::Separator();
            ImGui::Text("Cross swell (breaks up the crest lines)");
            ImGui::SliderFloat("Cross Period", &layerParams.crossWavePeriod, 2.0f, 250.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Cross train wavelength (world units).");
            ImGui::SliderFloat("Cross Speed", &layerParams.crossWaveSpeed, 0.0f, 30.0f, "%.2f");
            ImGui::SliderFloat("Cross Amplitude", &layerParams.crossWaveAmplitude, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Cross Phase Offset", &layerParams.crossWavePhase, -256.0f, 256.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Offset of the cross train along the shore direction (world units).\n"
                                              "Both trains move along the same shore direction; this shifts them apart.");
            ImGui::Separator();
            ImGui::SliderFloat("Chop Amount", &layerParams.waveChopAmount, 0.0f, 2.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("FBM chop mixed into the ridged crests (uses the Noise Detail spectrum).");
            ImGui::SliderFloat("Ridge Stretch", &layerParams.waveRidgeStretch, 1.0f, 24.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Anisotropy of the ridged Perlin crests: along/across frequency ratio.\n"
                                              "Higher = long, wave-like crest lines; 1 = isotropic ridge blobs.");
            ImGui::SliderFloat("Crest Phase Warp", &layerParams.waveWarpAmount, 0.0f, 3.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Extra Perlin domain-warp drift of the ridged crests (feature units).\n"
                                              "Uses the same Noise Detail spectrum as the chop.");
            ImGui::SliderFloat("Crest Amp Variation", &layerParams.waveAmpVariation, 0.0f, 0.95f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Perlin-driven local crest height variation (0 = uniform crests).");
            ImGui::SliderFloat("Whitecap Onset", &layerParams.whitecapOnset, 0.0f, 0.95f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Shoaling progress at which whitecaps start to appear (0..1).");
        }

        // ── Depth zones and wave shape ──
        if (ImGui::CollapsingHeader("Wave Zones (Thickness)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Waves are shaped by the measured water thickness: deep ocean swell above "
                               "Zone Deep, shoaling and breakers toward Zone Break, foam and a decaying "
                               "line wave below it, ending at the waterline.");
            ImGui::SliderFloat("Zone Deep Depth", &layerParams.zoneDeepDepth, 1.0f, 1024.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Water thickness at/above which the open-ocean swell is at full strength.");
            ImGui::SliderFloat("Zone Break Depth", &layerParams.zoneBreakDepth, 1.0f, 512.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Breaker line: waves crash and foam is born around this thickness.");
            ImGui::SliderFloat("Zone Shallow Depth", &layerParams.zoneShallowDepth, 0.0f, 256.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Below this thickness only the residual shore line wave remains.");
            ImGui::Separator();
            ImGui::SliderFloat("Shoal Gain", &layerParams.waveShoalGain, 0.0f, 4.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Amplitude gain from the deep zone toward the breaker line.");
            ImGui::SliderFloat("Shoal Speed Drop", &layerParams.waveShoalSpeed, 0.0f, 1.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("How much the phase speed drops as the water shallows [0..1].");
            ImGui::SliderFloat("Shallow Decay", &layerParams.waveShallowDecay, 0.05f, 6.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Amplitude decay exponent between the breaker line and the shallow zone.");
            ImGui::SliderFloat("Line Wave Amplitude", &layerParams.waveLineAmplitude, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Residual shore line wave height fraction below the shallow zone.");
            ImGui::SliderFloat("Height Falloff", &layerParams.waveHeightFalloff, 0.0f, 4.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Global depth taper of the wave height for ALL waves:\n"
                                              "pow(depth / Zone Deep, falloff), so the height decreases\n"
                                              "from full in the deep zone to 0 at the waterline.\n"
                                              "0 = disabled (the zone envelope alone shapes the height).");
            ImGui::SliderFloat("Breaker Amplitude", &layerParams.breakerAmplitude, 0.0f, 3.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Extra crest height concentrated at the breaker line.");
            ImGui::SliderFloat("Breaker Width", &layerParams.breakerWidth, 0.1f, 128.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Depth half-width of the breaker amplitude bump.");
            ImGui::SliderFloat("Breaker Curl", &layerParams.breakerCurl, -0.9f, 0.9f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Forward-leaning lip of the breaking crest (profile skew).\n"
                                              "Sign flips the lean direction, 0 = symmetric crest.");
            ImGui::SliderFloat("Crest Curvature", &layerParams.breakerCrestCurve, 0.0f, 2.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Hooks the breaking crest line into a curl around the break line\n"
                                              "(0 = straight crests). Only active where the wave is breaking.");
            ImGui::Separator();
            ImGui::SliderFloat("Sharpness Deep", &layerParams.waveSharpDeep, 0.25f, 12.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Crest sharpness in deep water (1 = cosine, higher = peaked).");
            ImGui::SliderFloat("Sharpness Break", &layerParams.waveSharpBreak, 0.25f, 16.0f, "%.2f");
            ImGui::SliderFloat("Sharpness Shallow", &layerParams.waveSharpShallow, 0.25f, 12.0f, "%.2f");
        }

        // ── Organic calm patches ──
        if (ImGui::CollapsingHeader("Organic Mask")) {
            ImGui::TextWrapped("Low-frequency noise mask: in calm patches the wave amplitude can drop to zero.");
            ImGui::SliderFloat("Mask Period", &layerParams.waveMaskPeriod, 10.0f, 2000.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Calm-patch feature period (world units). Larger = broader patches.");
            ImGui::SliderFloat("Mask Threshold", &layerParams.waveMaskThreshold, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Mask Softness", &layerParams.waveMaskSoftness, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Mask Speed", &layerParams.waveMaskSpeed, 0.0f, 1.0f, "%.3f");
        }

        // ── Foam / whitewater ──
        if (ImGui::CollapsingHeader("Foam", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Foam", &layerParams.enableFoam);
            ImGui::ColorEdit3("Foam Color", &layerParams.foamColor.x);
            ImGui::SliderFloat("Foam Amount", &layerParams.foamColorAmount, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Crest Threshold", &layerParams.foamCrestThreshold, 0.0f, 0.99f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Wave crest height at which foam starts appearing.");
            ImGui::SliderFloat("Trail Lag", &layerParams.foamTrailPhase, 0.0f, 2.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Lag of the trailing foam band behind the ridged crest\n"
                                              "(ridge-feature units; 1 = one crest feature).");
            ImGui::SliderFloat("Foam Edge", &layerParams.foamEdge, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Hardness of the foam mask edges.\n"
                                              "0 = soft gradients, 1 = hard, well-defined foam edges.");
            ImGui::SliderFloat("Foam Coverage", &layerParams.foamCoverage, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Global foam coverage multiplier (lighter/airier foam < 1).");
            ImGui::SliderFloat("Foam Shore Speed", &layerParams.foamShoreSpeed, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Foam advection speed factor at the shoreline (1 = same as\n"
                                              "the breaker). Foam races off the curl and slows near shore.");
            ImGui::SliderFloat("Foam Lag Growth", &layerParams.foamLagGrowth, 0.0f, 6.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("How much the trailing foam falls behind the lip as the wave\n"
                                              "approaches the shore.");
            ImGui::SliderFloat("Foam Decay", &layerParams.foamDecay, 0.0f, 0.5f, "%.4f");
            ImGuiHelpers::SetTooltipIfHovered("Foam extinction per meter below the breaker line.");
            ImGui::SliderFloat("Shore Foam", &layerParams.foamShoreAmount, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Persistent foam line on the shallow band (fades with the line wave).");
            ImGui::SliderFloat("Contact Width", &layerParams.foamContactWidth, 0.05f, 32.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Depth band of the final foam line where the water meets the\n"
                                              "solid (peaks at depth 0, world units).");
            ImGui::SliderFloat("Contact Amount", &layerParams.foamContactAmount, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Contact Opacity", &layerParams.foamContactAlpha, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Minimum composite opacity forced for the contact line so the\n"
                                              "last water pixels still render foam.");
            ImGui::SliderFloat("Contact Pulse Floor", &layerParams.foamContactFloor, 0.0f, 1.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Residual contact-foam strength between incoming crests.\n"
                                              "0 = the line fully retreats and arrives with each wave,\n"
                                              "1 = continuous line.");
            ImGui::Separator();
            ImGui::SliderFloat("Foam Noise Period", &layerParams.foamNoisePeriod, 1.0f, 200.0f, "%.2f");
            ImGuiHelpers::SetTooltipIfHovered("Foam breakup feature period (world units).");
            ImGui::SliderFloat("Foam Noise Speed", &layerParams.foamNoiseSpeed, 0.0f, 2.0f, "%.3f");
            ImGui::SliderFloat("Foam Noise Amount", &layerParams.foamNoiseAmount, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Foam Mask Floor", &layerParams.foamMaskFloor, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Foam Diffuse Floor", &layerParams.foamDiffuseFloor, 0.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("Foam Ambient", &layerParams.foamAmbient, 0.0f, 1.0f, "%.3f");
        }

        // ── Noise detail (chop + refraction + specular) ──
        if (ImGui::CollapsingHeader("Noise Detail")) {
            ImGui::SliderFloat("Noise Period", &layerParams.noisePeriod, 0.25f, 100.0f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Chop/refraction/specular noise feature period (world units).\n"
                                              "Larger = broader noise, 0 = disabled.");
            ImGui::SliderInt("Noise Octaves", &layerParams.noiseOctaves, 1, 8);
            ImGui::SliderFloat("Noise Persistence", &layerParams.noisePersistence, 0.1f, 0.9f);
            ImGui::SliderFloat("Noise Lacunarity", &layerParams.noiseLacunarity, 1.0f, 4.0f);
            ImGui::SliderFloat("Noise Time Speed", &layerParams.noiseTimeSpeed, 0.0f, 5.0f);
        }

        // ── Volumetric scattering ──
        if (ImGui::CollapsingHeader("Volumetric Scattering", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Single-scattering sunlight inside the measured water column "
                               "(Henyey-Greenstein phase, saturating with thickness).");
            ImGui::Checkbox("Enable Volumetric", &layerParams.enableVolumetric);
            ImGui::ColorEdit3("Scatter Color", &layerParams.volumetricColor.x);
            ImGui::SliderFloat("Scatter Strength", &layerParams.volumetricStrength, 0.0f, 2.0f, "%.3f");
            ImGui::SliderFloat("Scatter Density", &layerParams.volumetricDensity, 0.0f, 2.0f, "%.4f");
            ImGui::SliderFloat("Scatter Anisotropy", &layerParams.volumetricPhaseG, -0.95f, 0.95f, "%.3f");
            ImGuiHelpers::SetTooltipIfHovered("Henyey-Greenstein g: 0 = isotropic, >0 = forward scattering (sun halo).");
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
            ImGui::Separator();
            ImGui::ColorEdit3("Ocean Color", &layerParams.oceanColor.x);
            ImGuiHelpers::SetTooltipIfHovered("Third color stop: deep-ocean tint beyond Ocean Color Start.");
            ImGui::SliderFloat("Ocean Color Start", &layerParams.oceanColorStart, 0.0f, 512.0f, "%.1f");
            ImGui::SliderFloat("Ocean Depth Scale", &layerParams.oceanDepthScale, 1.0f, 512.0f, "%.1f");
            ImGuiHelpers::SetTooltipIfHovered("Thickness ramp over which the deep tint blends to the ocean color.");
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
