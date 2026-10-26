#include "PhysicsWidget.hpp"

#include "components/ColumnLayout.hpp"

#include <vector>

PhysicsWidget::PhysicsWidget(Settings& settingsRef)
    : Widget("Physics", u8"\uf0c3"), settings(settingsRef) {}

void PhysicsWidget::render() {
    // Same window setup as the Settings widget: resizable, horizontal scrollbar
    // only for column overflow, mouse wheel never scrolls vertically.
    auto wg = ImGuiComponents::BeginColumnWindow(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // Sections match the Settings widget's structure: each is self-contained
    // (header + controls) and flows into fixed-width columns by height.
    std::vector<ImGuiComponents::ColumnSection> sections;
    sections.reserve(5);

    // 0: Master
    sections.push_back({[this]() {
        ImGui::Text("Master");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::CheckboxField("Droplets Enabled", &settings.dropletsEnabled,
            "Master toggle for the GPU water droplets (instanced proxy cubes,\n"
            "SDF ray-marched refraction/reflection shading).");
        ImGuiComponents::SliderIntField("Count", &settings.dropletCount, 128, 8192, "%d",
            "Simulated droplets drawn per frame (128..8192).\nHigher = denser spray, more GPU cost.");
        if (ImGui::Button("Reset now", ImVec2(ImGuiComponents::kColumnWidth, 0.0f))) {
            resetRequested = true;
        }
        ImGuiComponents::TooltipOnHover("Re-seed all droplets at the spawn shell/height immediately.\n"
            "The app reads and clears the request on the next frame.");
        ImGuiComponents::SliderFloatField("Auto Reset (s)", &settings.dropletResetSeconds, 5.0f, 300.0f, "%.0f",
            "Seconds between automatic re-seeds (5..300).");
    }});

    // 1: Spawning
    sections.push_back({[this]() {
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
    }});

    // 2: Physics
    sections.push_back({[this]() {
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
    }});

    // 3: Optics
    sections.push_back({[this]() {
        ImGui::Text("Optics");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::SliderFloatField("IOR", &settings.dropletIor, 1.0f, 2.0f, "%.3f",
            "Index of refraction (1.0 = no bending, 1.333 = water, 2.0 = dense glass).");
    }});

    // 4: Diagnostics
    sections.push_back({[this]() {
        ImGui::Text("Diagnostics");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::CheckboxField("Stats Readback", &settings.dropletStats,
            "Read back GPU droplet stats (active count, solver cost) for the debug overlay.");
    }});

    static std::vector<float> cachedH;
    const std::vector<float> estimates = {
        ImGuiComponents::EstimateSectionHeight(4), // Master
        ImGuiComponents::EstimateSectionHeight(4), // Spawning
        ImGuiComponents::EstimateSectionHeight(6), // Physics
        ImGuiComponents::EstimateSectionHeight(2), // Optics
        ImGuiComponents::EstimateSectionHeight(2), // Diagnostics
    };
    ImGuiComponents::LayoutSections(sections, cachedH, estimates);
}
