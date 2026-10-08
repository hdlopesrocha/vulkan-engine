#include "GrassWidget.hpp"

#include <imgui.h>

#include "../vulkan/renderer/sdf/SdfRenderer.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ColumnLayout.hpp"

GrassWidget::GrassWidget(SdfRenderer* sdfRenderer_)
    : Widget("Grass", u8"\uf4d8"), sdfRenderer(sdfRenderer_) {
    isOpen = false;
}

void GrassWidget::render() {
    if (!sdfRenderer) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    SdfEffectConfig& cfg = sdfRenderer->config();
    SdfGrassConfig& grass = cfg.grass;
    const SdfStats& stats = sdfRenderer->getStats();

    ImGui::TextWrapped("SDF grass clumps derived from the existing vegetation "
                       "instances (position, vegetation type/biome, smooth "
                       "normal). Each instance becomes one grouped clump that "
                       "the SDF raymarcher expands into many procedural blades; "
                       "the aggregate envelope skips empty space by SDF "
                       "distance.");
    ImGui::Spacing();
    ImGui::Text("clumps: %u   vegetation chunks: %u   SDF containers: %u",
                stats.grassAnchors, stats.grassChunks, stats.containerCount);
    ImGui::Spacing();

    // 1: Placement (collector caps; density stays owned by the vegetation).
    if (ImGui::CollapsingHeader("Placement", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool enabled = grass.enabled;
        if (ImGuiComponents::CheckboxField("Grass enabled", &enabled,
                "Render grass clumps in the generic SDF pass. The streamed "
                "vegetation instances stay collected while disabled.")) {
            sdfRenderer->setGrassEnabled(enabled);
        }
        int cap = grass.maxPerChunk;
        if (ImGuiComponents::SliderIntField("Max per chunk", &cap, 0, 1024, "%d",
                "Clump cap per vegetation chunk (applies to newly streamed "
                "chunks; the vegetation density itself is unchanged).")) {
            sdfRenderer->setGrassMaxPerChunk(cap);
        }
    }

    // 2: Clump shape (packed into the Grass definitions at rebuild).
    if (ImGui::CollapsingHeader("Shape", ImGuiTreeNodeFlags_DefaultOpen)) {
        float radius = grass.clumpRadius;
        if (ImGuiComponents::SliderFloatField("Clump radius", &radius, 0.05f, 2.0f, "%.2f",
                "Local blade-root radius (1 = the per-instance vegetation scale).")) {
            sdfRenderer->setGrassClumpRadius(radius);
        }
        float height = grass.bladeHeight;
        if (ImGuiComponents::SliderFloatField("Blade height", &height, 0.05f, 4.0f, "%.2f",
                "Local blade height (1 = the per-instance vegetation scale).")) {
            sdfRenderer->setGrassBladeHeight(height);
        }
        float width = grass.bladeWidth;
        if (ImGuiComponents::SliderFloatField("Blade width", &width, 0.005f, 0.25f, "%.3f",
                "Local blade half-width at the base.")) {
            sdfRenderer->setGrassBladeWidth(width);
        }
        int count = grass.bladeCount;
        if (ImGuiComponents::SliderIntField("Blades / clump", &count, 1, 64, "%d",
                "Blades evaluated per clump; the shader reduces this with "
                "distance while the aggregate envelope takes over.")) {
            sdfRenderer->setGrassBladeCount(count);
        }
        float curve = grass.curvature;
        if (ImGuiComponents::SliderFloatField("Curvature", &curve, 0.0f, 1.5f, "%.2f",
                "Sideways tip offset as a fraction of the blade height.")) {
            sdfRenderer->setGrassCurvature(curve);
        }
        float tip = grass.tipWidth;
        if (ImGuiComponents::SliderFloatField("Tip width", &tip, 0.05f, 1.0f, "%.2f",
                "Blade tip radius as a fraction of the base width.")) {
            sdfRenderer->setGrassTipWidth(tip);
        }
    }

    // 3: Wind (applied as a rigid lean from the shared wind field).
    if (ImGui::CollapsingHeader("Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
        float gain = grass.windGain;
        if (ImGuiComponents::SliderFloatField("Wind gain", &gain, 0.0f, 1.0f, "%.3f",
                "Lean radians per m/s of the shared wind field at the clump.")) {
            sdfRenderer->setGrassWindGain(gain);
        }
        float lean = grass.maxLean;
        if (ImGuiComponents::SliderFloatField("Max lean", &lean, 0.0f, 1.4f, "%.2f",
                "Clamp on the wind lean (radians); also pads the CPU bounds.")) {
            sdfRenderer->setGrassMaxLean(lean);
        }
    }

    // 4: Material.
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        float roughness = grass.roughness;
        if (ImGuiComponents::SliderFloatField("Roughness", &roughness, 0.02f, 1.0f, "%.2f",
                "Surface roughness of the blade shading.")) {
            sdfRenderer->setGrassRoughness(roughness);
        }
        float tint[3] = {grass.tint.x, grass.tint.y, grass.tint.z};
        if (ImGuiComponents::ColorEdit3Field("Tint", tint,
                "Multiplies the per-vegetation-type blade colors.")) {
            sdfRenderer->setGrassTint(glm::vec3(tint[0], tint[1], tint[2]));
        }
    }

    // 5: Shadow casting (grass-only EVSM caster).
    if (ImGui::CollapsingHeader("Shadows")) {
        float lod = grass.shadowLodScale;
        if (ImGuiComponents::SliderFloatField("Shadow LOD", &lod, 4.0f, 120.0f, "%.0f",
                "Camera-scale value used by the grass shadow march: 4-16 keeps "
                "full blades, ~32 uses the reduced blade set, >=60 uses the "
                "aggregate envelope only (cheapest).")) {
            sdfRenderer->setGrassShadowLodScale(lod);
        }
    }

    ImGui::Spacing();
    if (ImGui::Button("Rebuild grass")) {
        sdfRenderer->markGrassDirty();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear grass")) {
        sdfRenderer->clearGrass();
    }
    ImGuiComponents::TooltipOnHover(
        "Rebuild re-flattens the retained vegetation-derived clumps; Clear "
        "drops every tracked chunk (re-streaming vegetation re-adds theirs).");
}
