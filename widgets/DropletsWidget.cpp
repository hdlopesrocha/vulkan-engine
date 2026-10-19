#include "DropletsWidget.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ColumnLayout.hpp"
#include <vector>
#include <functional>

DropletsWidget::DropletsWidget(Settings& settingsRef)
    : Widget("Droplets", u8"\uf043"), settings(settingsRef) {}

namespace {
// Every column is exactly this wide; separators match it 1:1.
// Same value as SettingsWidget so both windows pack identically.
constexpr float kSettingsColWidth = 256.0f;
} // namespace

void DropletsWidget::render() {
    // Same window setup as Settings: resizable, horizontal scrollbar only for
    // column overflow, mouse wheel never scrolls vertically.
    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(800, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    // ---- Sections: each must be self-contained (header + controls) so it can
    // ---- be moved as a whole into another column when the current one fills up.
    std::vector<std::function<void()>> sections;
    sections.reserve(5);

    // 0: Master
    sections.emplace_back([this]() {
        ImGui::Text("Master");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::CheckboxField("Droplets Enabled", &settings.dropletsEnabled,
            "Master toggle for the GPU water droplets (instanced proxy cubes,\n"
            "SDF ray-marched refraction/reflection shading).");
        ImGuiComponents::SliderIntField("Count", &settings.dropletCount, 128, 8192, "%d",
            "Simulated droplets drawn per frame (128..8192).\nHigher = denser spray, more GPU cost.");
        if (ImGui::Button("Reset now", ImVec2(kSettingsColWidth, 0.0f))) {
            resetRequested = true;
        }
        ImGuiComponents::TooltipOnHover("Re-seed all droplets at the spawn shell/height immediately.\n"
            "The app reads and clears the request on the next frame.");
        ImGuiComponents::SliderFloatField("Auto Reset (s)", &settings.dropletResetSeconds, 5.0f, 300.0f, "%.0f",
            "Seconds between automatic re-seeds (5..300).");
    });

    // 1: Spawning
    sections.emplace_back([this]() {
        ImGui::Text("Spawning");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::SliderFloatField("Spawn Radius (m)", &settings.dropletSpawnRadius, 16.0f, 512.0f, "%.0f",
            "Radius of the spawn volume centred at (0, Spawn Height, 0) (16..512 m).\nAll droplets are re-seeded inside this sphere.");
        ImGuiComponents::SliderFloatField("Spawn Height Above Terrain (m)", &settings.dropletSpawnHeight,
            5.0f, 2000.0f, "%.0f",
            "Height of the spawn centre above the height map (5..2000 m). The\n"
            "droplets are re-seeded in a ball here and fall onto/into the terrain.");
        ImGuiComponents::SliderFloatField("Droplet Radius (m)", &settings.dropletRadius, 0.2f, 8.0f, "%.2f",
            "Particle sphere radius (0.2..8 m): the same value drives the SDF\ncollision query and the ray-marched render — physics and rendering never\ndiverge on the collision shape.");
    });

    // 2: Physics
    sections.emplace_back([this]() {
        ImGui::Text("Physics");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::SliderFloatField("Gravity (m/s^2)", &settings.dropletGravity, 0.0f, 30.0f, "%.1f",
            "Downward acceleration (0..30 m/s^2).");
        ImGuiComponents::SliderFloatField("Restitution", &settings.dropletRestitution, 0.0f, 1.0f, "%.2f",
            "Bounce response on contact (0 = absorb, 1 = perfectly elastic).");
        ImGuiComponents::SliderFloatField("Friction", &settings.dropletFriction, 0.0f, 1.0f, "%.2f",
            "Tangential velocity loss on contact (0 = frictionless).");
        ImGuiComponents::SliderFloatField("Damping", &settings.dropletDamping, 0.0f, 1.0f, "%.2f",
            "Air drag per step (0 = none).");
        ImGuiComponents::SliderIntField("Solver Iterations", &settings.dropletSolverIterations, 1, 8, "%d",
            "Constraint solver iterations per step (1..8).\nHigher = stiffer contacts, more GPU cost.");
        ImGuiComponents::SliderIntField("Substeps", &settings.dropletSubsteps, 1, 8, "%d",
            "Simulation substeps per frame (1..8).\nHigher = more stable at speed, more GPU cost.");
    });

    // 3: Optics
    sections.emplace_back([this]() {
        ImGui::Text("Optics");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::SliderFloatField("IOR", &settings.dropletIor, 1.0f, 2.0f, "%.3f",
            "Index of refraction (1.0 = no bending, 1.333 = water, 2.0 = dense glass).");
    });

    // 4: Diagnostics
    sections.emplace_back([this]() {
        ImGui::Text("Diagnostics");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::CheckboxField("Stats Readback", &settings.dropletStats,
            "Read back GPU droplet stats (active count, solver cost) for the debug overlay.");
    });

    // ---- Packing: best-fit over the last frame's measured section heights.
    // ---- Sections are visited in importance order; each goes into the fullest
    // ---- column that still fits it, so leftover gaps get filled instead of
    // ---- stranding empty column space. New columns open only when nothing
    // ---- fits; overflow scrolls horizontally (fixed 256px columns).
    const int n = static_cast<int>(sections.size());
    static std::vector<float> cachedH;
    if (static_cast<int>(cachedH.size()) != n) {
        // First frame estimates; refined by measurement from frame 2 on.
        cachedH.assign(n, 170.0f);
        cachedH[0] = 260.0f; // Master
        cachedH[1] = 220.0f; // Spawning
        cachedH[2] = 360.0f; // Physics
        cachedH[3] = 150.0f; // Optics
        cachedH[4] = 150.0f; // Diagnostics
    }

    float availW = ImGui::GetContentRegionAvail().x;
    float availH = ImGui::GetContentRegionAvail().y;
    if (availW < 50.0f) availW = 300.0f;
    if (availH < 50.0f) availH = 500.0f;
    // Safety margin so rounding/trailing spacing can never push content 1px
    // past the bottom edge (which would summon a vertical scrollbar).
    // Columns are sized against flowH, guaranteeing content height < availH.
    const float flowH = availH - 4.0f;

    // No width cap: as many columns as the height requires. Horizontal
    // scrolling handles overflow instead of squeezing columns.
    constexpr float kSectionGap = 8.0f;
    std::vector<int> colOf(n, 0);
    std::vector<float> colH(1, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float h = cachedH[i] > 1.0f ? cachedH[i] : 170.0f;
        // Tightest column that fits wins; empty columns always accept so a
        // section taller than the window still lands somewhere sane.
        int best = -1;
        float bestRem = 1e30f;
        for (int c = 0; c < static_cast<int>(colH.size()); ++c) {
            const float rem = flowH - colH[c];
            if ((colH[c] <= 0.0f || rem >= h) && rem - h < bestRem) {
                best = c;
                bestRem = rem - h;
            }
        }
        if (best < 0) {
            best = static_cast<int>(colH.size());
            colH.emplace_back(0.0f);
        }
        colOf[i] = best;
        colH[best] += h + kSectionGap;
    }
    const int nCols = static_cast<int>(colH.size());

    auto renderSectionMeasured = [&](int idx, bool firstInColumn) {
        if (!firstInColumn) {
            ImGui::Spacing();
            ImGuiComponents::ColSeparator();
            ImGui::Spacing();
        }
        const float y0 = ImGui::GetCursorScreenPos().y;
        sections[idx]();
        const float y1 = ImGui::GetCursorScreenPos().y;
        const float measured = y1 - y0;
        if (measured > 1.0f) cachedH[idx] = measured;
    };

    // Fixed column width: every column is exactly kSettingsColWidth; overflow scrolls.
    constexpr float colWidth = kSettingsColWidth;
    const float spacingX = ImGui::GetStyle().ItemSpacing.x;

    for (int c = 0; c < nCols; ++c) {
        if (c > 0) ImGui::SameLine(0.0f, spacingX);
        ImGui::BeginGroup();
        // Enforce the column width so TextWrapped + full-width widgets wrap
        // consistently and the group extends past the window edge when needed.
        ImGui::Dummy(ImVec2(colWidth, 0.0f));
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + colWidth);
        bool first = true;
        for (int i = 0; i < n; ++i) {
            if (colOf[i] != c) continue;
            renderSectionMeasured(i, first);
            first = false;
        }
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }
}
