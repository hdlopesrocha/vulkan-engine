#include "RocksWidget.hpp"

#include <algorithm>
#include <cmath>
#include <imgui.h>

#include "../vulkan/renderer/sdf/SdfRenderer.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ColumnLayout.hpp"

RocksWidget::RocksWidget(SdfRenderer* sdfRenderer_)
    : Widget("Rocks", u8"\uf1b3"), sdfRenderer(sdfRenderer_) {
    isOpen = false;
}

void RocksWidget::render() {
    if (!sdfRenderer) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    SdfEffectConfig& cfg = sdfRenderer->config();
    SdfRockConfig& rocks = cfg.rocks;
    const SdfStats& stats = sdfRenderer->getStats();

    ImGui::TextWrapped("SDF boulders spawned from brush-7 rock terrain. "
                       "One rock per spacing x spacing m of rock surface; "
                       "shape, size and material rebuild live.");
    ImGui::Spacing();
    ImGui::Text("rocks: %u   rock chunks: %u   SDF containers: %u",
                stats.rockAnchors, stats.rockChunks, stats.containerCount);
    ImGui::Spacing();

    // 1: Placement (density + caps).
    if (ImGui::CollapsingHeader("Placement", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool enabled = rocks.enabled;
        if (ImGuiComponents::CheckboxField("Rocks enabled", &enabled,
                "Show boulders. Candidates stay collected while disabled.")) {
            sdfRenderer->setRocksEnabled(enabled);
        }
        float spacing = rocks.spacing;
        if (ImGuiComponents::SliderFloatField("Spacing (m)", &spacing,
                rocks.minSpacing, 4096.0f, "%.0f",
                "One rock per spacing x spacing metres of rock surface "
                "(default 512 = one per 512 x 512 m).",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setRockSpacing(spacing);
        }
        int cap = rocks.maxPerChunk;
        if (ImGuiComponents::SliderIntField("Max per chunk", &cap, 0, 256, "%d",
                "Candidate cap per terrain chunk (density is applied on top "
                "at rebuild).")) {
            sdfRenderer->setRockMaxPerChunk(cap);
        }
    }

    // 2: Size + placement on the surface.
    if (ImGui::CollapsingHeader("Size", ImGuiTreeNodeFlags_DefaultOpen)) {
        float scale = rocks.scale;
        if (ImGuiComponents::SliderFloatField("Scale / radius (m)", &scale, 4.0f, 256.0f, "%.0f",
                "Boulder scale = sphere radius in world metres (64 = 64 m).",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setRockScale(scale);
        }
        float variation = rocks.scaleVariation;
        if (ImGuiComponents::SliderFloatField("Size variation", &variation, 0.0f, 1.0f, "%.2f",
                "Random +/- fraction of the radius.")) {
            sdfRenderer->setRockScaleVariation(variation);
        }
        float embed = rocks.embed;
        if (ImGuiComponents::SliderFloatField("Embed", &embed, 0.0f, 0.95f, "%.2f",
                "Fraction of the radius sunk below the surface (0 = tangent).")) {
            sdfRenderer->setRockEmbed(embed);
        }
    }

    // 3: Shape (Perlin displacement of the sphere).
    if (ImGui::CollapsingHeader("Shape", ImGuiTreeNodeFlags_DefaultOpen)) {
        float noiseScale = rocks.noiseScale;
        if (ImGuiComponents::SliderFloatField("Noise scale", &noiseScale, 0.5f, 8.0f, "%.2f",
                "Perlin frequency per local unit (features shrink as it rises; "
                "the noise scales with the boulder, not world metres).",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setRockNoiseScale(noiseScale);
        }
        float noiseAmp = rocks.noiseAmplitude;
        if (ImGuiComponents::SliderFloatField("Noise amplitude", &noiseAmp, 0.0f, 1.0f, "%.2f",
                "Displacement as a fraction of the radius (0 = plain sphere).")) {
            sdfRenderer->setRockNoiseAmplitude(noiseAmp);
        }
    }

    // 4: Surface material (scene texture arrays + PBR scalars).
    if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
        int layer = static_cast<int>(std::lround(rocks.textureLayer));
        if (ImGuiComponents::SliderIntField("Texture layer", &layer, -1, 31, "%d",
                "Scene texture-array layer for albedo/normal (7 = rock). "
                "-1 disables texturing (flat base color).")) {
            sdfRenderer->setRockTextureLayer(static_cast<float>(layer));
        }
        float tiling = rocks.textureTiling;
        if (ImGuiComponents::SliderFloatField("Texture tiling (m)", &tiling, 1.0f, 512.0f, "%.0f",
                "World metres per texture repeat.",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setRockTextureTiling(tiling);
        }
        float roughness = rocks.roughness;
        if (ImGuiComponents::SliderFloatField("Roughness", &roughness, 0.02f, 1.0f, "%.2f",
                "Surface roughness (drives the specular lobe).")) {
            sdfRenderer->setRockRoughness(roughness);
        }
        float metallic = rocks.metallic;
        if (ImGuiComponents::SliderFloatField("Metallic", &metallic, 0.0f, 1.0f, "%.2f",
                "Metallic response of the specular lobe.")) {
            sdfRenderer->setRockMetallic(metallic);
        }
        float tint[3] = {rocks.tint.x, rocks.tint.y, rocks.tint.z};
        if (ImGuiComponents::ColorEdit3Field("Tint", tint,
                "Multiplies the sampled albedo (1,1,1 = unmodified).")) {
            sdfRenderer->setRockTint(glm::vec3(tint[0], tint[1], tint[2]));
        }
    }

    ImGui::Spacing();
    if (ImGui::Button("Rebuild rocks")) {
        sdfRenderer->markRocksDirty();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear rocks")) {
        sdfRenderer->clearRocks();
    }
    ImGuiComponents::TooltipOnHover(
        "Rebuild re-flattens the retained candidates; Clear drops every "
        "collected candidate (streamed chunks re-add theirs when re-published).");
}
