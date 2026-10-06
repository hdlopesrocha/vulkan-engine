#include "RaymarchWidget.hpp"

#include <imgui.h>
#include <functional>
#include <vector>

#include "../math/Camera.hpp"
#include "components/ImGuiHelpers.hpp"
#include "components/ColumnLayout.hpp"

RaymarchWidget::RaymarchWidget(SdfRenderer* sdf, Camera* cam, VegetationRenderer* veg)
    : Widget("Ray Marching"), sdfRenderer(sdf), camera(cam), vegetationRenderer(veg) {
    isOpen = false;
}

namespace {
// Column width matches the Settings menu (shared helper default).
constexpr float kRaymarchColWidth = ImGuiComponents::kColumnWidth;
} // namespace

void RaymarchWidget::render() {
    if (!sdfRenderer) return;
    ImGui::SetNextWindowSize(ImVec2(640, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    // All tuning below binds per-frame locals read from the shared config
    // (SdfRenderer owns the only stored copy); edits commit through the
    // renderer setters, which clamp and route uploads/rebuilds. Fire and
    // smoke are both just shapes in that generic scene.

    // ---- Sections: each is self-contained (header + controls) so the
    // ---- column packer below can move it as a whole, exactly like the
    // ---- Settings menu.
    std::vector<std::function<void()>> sections;
    sections.reserve(10);

    // 0: Raymarcher (generic traversal + shading controls).
    sections.emplace_back([this]() {
        ImGui::Text("Raymarcher");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Generic GPU SDF raymarcher (surface/volume/emissive/transparent). "
                           "Traversal, grid and shading are shared by every SDF shape.");
        SdfEffectConfig& cfg = sdfRenderer->config();
        (void)cfg;
        if (ImGuiComponents::SliderFloatField("Time scale", &timeScale, 0.0f, 4.0f, "%.2f",
                "Global SDF time multiplier.")) {
            // live: read every frame by the SDF task
        }
        static int mode = 1;
        ImGui::RadioButton("Surface##raymarch", &mode, 0);
        ImGui::SameLine(); ImGui::RadioButton("Volume##raymarch", &mode, 1);
        ImGui::SameLine(); ImGui::RadioButton("Emissive##raymarch", &mode, 2);
        ImGui::SameLine(); ImGui::RadioButton("Transparent##raymarch", &mode, 3);
        sdfRenderer->setRenderMode(static_cast<SdfRenderer::RenderMode>(mode));
        // Debug view: bit 0 = on, bits 1-2 = Steps / Candidates / Field.
        // Candidates BLACK = no SDF ever evaluated along the ray.
        // Field WHITE = ray passed close to a surface.
        static int dbg = 0;
        ImGui::RadioButton("Debug off##raymarch", &dbg, 0);
        ImGui::SameLine(); ImGui::RadioButton("Steps##raymarch", &dbg, 1);
        ImGui::SameLine(); ImGui::RadioButton("Candidates##raymarch", &dbg, 2);
        ImGui::SameLine(); ImGui::RadioButton("Field##raymarch", &dbg, 3);
        uint32_t flags = 0u;
        if (dbg == 1) flags = 1u;
        else if (dbg == 2) flags = 1u | (1u << 1u);
        else if (dbg == 3) flags = 1u | (2u << 1u);
        sdfRenderer->setDebugFlags(flags);
    });

    // 1: Fire shape (lava-anchored volumetric flames).
    sections.emplace_back([this]() {
        ImGui::Text("Fire");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Volumetric flames anchored to brush-4 lava terrain as chunks stream in.");
        SdfEffectConfig& cfg = sdfRenderer->config();
        float v = 0.0f;
        // Display-only conversion: the shared value is flames/m².
        float areaPerFlame = (cfg.lava.density > 0.0f) ? (1.0f / cfg.lava.density) : 10000.0f;
        if (ImGuiComponents::SliderFloatField("m² per flame", &areaPerFlame, 100.0f, 100000.0f, "%.0f",
                "Lava surface area per flame (default 1 per 10000 m²). Applies to newly streamed chunks.",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setLavaDensity(1.0f / std::max(areaPerFlame, 1.0f));
        }
        v = cfg.lava.scale;
        if (ImGuiComponents::SliderFloatField("Flame scale", &v, 0.25f, 1024.0f, "%.1f",
                "Flame size multiplier (log scale, up to 1024). Applies to newly streamed chunks.",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setLavaScale(v);
        }
        v = cfg.lava.spikiness;
        if (ImGuiComponents::SliderFloatField("Spikiness", &v, 0.0f, 1.5f, "%.2f",
                "Flame tongue amplitude (0 = smooth rounded capsule). Applies on next scene rebuild.")) {
            sdfRenderer->setLavaSpikiness(v);
        }
        v = cfg.lava.tipRadius;
        if (ImGuiComponents::SliderFloatField("Tip radius", &v, 0.01f, 32.0f, "%.2f",
                "Flame tip roundness, local units x flame scale (= meters at scale 32). 0.01 ~ sharp cone tip. Applies on next scene rebuild.",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setLavaTipRadius(v);
        }
        v = cfg.lava.baseRadius;
        if (ImGuiComponents::SliderFloatField("Base radius", &v, 0.05f, 32.0f, "%.2f",
                "Flame base width, local units x flame scale (= meters at scale 32, up to 1024 m). Applies on next scene rebuild.",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setLavaBaseRadius(v);
        }
        v = cfg.lava.height;
        if (ImGuiComponents::SliderFloatField("Height", &v, 0.5f, 10.0f, "%.2f",
                "Flame base-to-tip height in local units (x instance scale). Applies on next scene rebuild.")) {
            sdfRenderer->setLavaHeight(v);
        }
        v = cfg.lava.spikeFreq;
        if (ImGuiComponents::SliderFloatField("Spike freq", &v, 0.5f, 6.0f, "%.2f",
                "Tongue count around the flame axis. Applies on next scene rebuild.")) {
            sdfRenderer->setLavaSpikeFreq(v);
        }
        v = cfg.lava.flameDensity;
        if (ImGuiComponents::SliderFloatField("Flame density", &v, 0.05f, 1.5f, "%.2f",
                "Volumetric density (lower = more transparent, like real flames). Applies on next scene rebuild.")) {
            sdfRenderer->setLavaFlameDensity(v);
        }
        auto st = sdfRenderer->getStats();
        ImGui::Text("flames=%u chunks=%u containers=%u cells=%u draws=%u",
            st.lavaAnchors, st.lavaChunks,
            st.containerCount, st.gridCellCount, st.lastDrawInstances);
    });

    // 1: Shape (generic rig: cloud / sphere / cube + translation, scale,
    // rotation).
    sections.emplace_back([this]() {
        ImGui::Text("Shape");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Master shape of the smoke volume: Cloud billows, "
                           "Sphere/Cube are dense bodies. The cube's bounding "
                           "sphere equals the scale, and rotation turns the "
                           "envelope and the billow domain together.");
        SdfEffectConfig& cfg = sdfRenderer->config();
        {
            int shape = cfg.smoke.shape;
            const char* shapeItems[] = {"Cloud", "Sphere", "Cube"};
            ImGuiComponents::FieldLabel("Shape", "Cloud = billowy volume; Sphere/Cube = dense bodies.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Shape##smoke");
            if (ImGui::Combo("##v", &shape, shapeItems, 3)) {
                sdfRenderer->setSmokeShape(shape);
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
        ImGuiComponents::FieldLabel("Translation", "Shape center in world units (rebuilds the scene).");
        {
            glm::vec3 pos = cfg.smoke.pos;
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Translation##smoke");
            if (ImGui::DragFloat3("##v", &pos[0], 1.0f)) {
                sdfRenderer->setSmokePosition(pos);
            }
            ImGui::PopID();
            ImGuiComponents::TooltipOnHover("Shape center in world units (rebuilds the scene).");
        }
        if (camera && ImGui::Button("Go to shape (center camera)")) {
            camera->translate(cfg.smoke.pos - camera->getPosition());
        }
        ImGuiComponents::TooltipOnHover("Teleport the camera to the shape center.");
        ImGui::Spacing();
        float v = cfg.smoke.scale;
        if (ImGuiComponents::SliderFloatField("Scale", &v, 8.0f, 1024.0f, "%.0f",
                "Master scale in meters: smoke growth size, sphere radius, cube bounding radius (rebuilds the scene).",
                ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setSmokeScale(v);
        }
        ImGui::Spacing();
        {
            float yaw = cfg.smoke.yawDeg;
            float pitch = cfg.smoke.pitchDeg;
            float roll = cfg.smoke.rollDeg;
            bool rotChanged = false;
            rotChanged |= ImGuiComponents::SliderFloatField("Yaw", &yaw, -180.0f, 180.0f, "%.0f",
                "Rotation about Y, degrees.");
            rotChanged |= ImGuiComponents::SliderFloatField("Pitch", &pitch, -180.0f, 180.0f, "%.0f",
                "Rotation about X, degrees.");
            rotChanged |= ImGuiComponents::SliderFloatField("Roll", &roll, -180.0f, 180.0f, "%.0f",
                "Rotation about Z, degrees.");
            if (rotChanged) {
                sdfRenderer->setSmokeRotation(yaw, pitch, roll);
            }
        }
    });

    // 2: Smoke shape (procedural smoke-grenade sphere + bullet interaction).
    sections.emplace_back([this]() {
        ImGui::Text("Smoke");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Procedural smoke grenade: growing sphere, layered "
                           "noise, bullet tunnel + pressure wave + turbulent "
                           "wake with refill. Animation loops every loop duration.");
        SdfEffectConfig& cfg = sdfRenderer->config();
        bool enabled = cfg.smoke.enabled;
        if (ImGuiComponents::CheckboxField("Enable smoke", &enabled,
                "Master toggle for the smoke volume (rebuilds the scene).")) {
            sdfRenderer->setSmokeEnabled(enabled);
        }
        float v = cfg.smoke.growthDuration;
        if (ImGuiComponents::SliderFloatField("Growth time", &v, 0.5f, 60.0f, "%.1f",
                "Rapid expansion time in seconds; the auto bullet waits it out.")) {
            sdfRenderer->setSmokeGrowthDuration(v);
        }
        v = cfg.smoke.loopDuration;
        {
            ImGuiComponents::FieldLabel("Loop duration", "Smoke animation repeats every loop duration.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Loop duration##smoke");
            if (ImGui::SliderFloat("##v", &v, 2.0f, 120.0f, "%.0f")) {
                sdfRenderer->setSmokeLoopDuration(v);
            }
            ImGui::PopID();
            ImGuiComponents::TooltipOnHover("Smoke animation repeats every loop duration.");
        }
        v = cfg.smoke.dissipation;
        {
            ImGuiComponents::FieldLabel("Dissipation", "End-of-loop fade strength.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Dissipation##smoke");
            if (ImGui::SliderFloat("##v", &v, 0.0f, 2.0f, "%.2f")) {
                sdfRenderer->setSmokeDissipation(v);
            }
            ImGui::PopID();
        }
        v = cfg.smoke.density;
        if (ImGuiComponents::SliderFloatField("Density", &v, 0.05f, 2.0f, "%.2f",
                "Base material density (rebuilds scene).")) {
            sdfRenderer->setSmokeDensity(v);
        }
        v = cfg.smoke.noiseScale;
        if (ImGuiComponents::SliderFloatField("Noise scale", &v, 0.001f, 0.2f, "%.4f",
                "Noise feature scale (1/m).", ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setSmokeNoiseScale(v);
        }
        v = cfg.smoke.noiseStrength;
        if (ImGuiComponents::SliderFloatField("Noise strength", &v, 0.0f, 2.0f, "%.2f",
                "Noise contrast.")) {
            sdfRenderer->setSmokeNoiseStrength(v);
        }
        v = cfg.smoke.noiseWarp;
        if (ImGuiComponents::SliderFloatField("Noise warp", &v, 0.0f, 2.0f, "%.2f",
                "Domain warp amount.")) {
            sdfRenderer->setSmokeNoiseWarp(v);
        }
        {
            v = cfg.smoke.windSpeed;
            float wa = cfg.smoke.windAngleDeg;
            bool windChanged = false;
            windChanged |= ImGuiComponents::SliderFloatField("Wind speed", &v, 0.0f, 60.0f, "%.1f",
                "Wind advection speed (m/s).");
            windChanged |= ImGuiComponents::SliderFloatField("Wind angle", &wa, 0.0f, 360.0f, "%.0f",
                "Wind direction in the XZ plane, 0 = +X.");
            if (windChanged) {
                sdfRenderer->setSmokeWind(v, wa);
            }
        }
    });

    // 3: Bullet.
    sections.emplace_back([this]() {
        ImGui::Text("Bullet");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        float br = cfg.bullet.radius;
        float bfr = cfg.bullet.finalRadius;
        float bs = cfg.bullet.speed;
        float bl = cfg.bullet.length;
        float ba = cfg.bullet.angleDeg;
        float bld = cfg.bullet.loopDuration;
        bool bChanged = false;
        bChanged |= ImGuiComponents::SliderFloatField("Bullet radius", &br, 0.5f, 100.0f, "%.1f",
            "Initial radius at launch (m); the tunnel tapers to the final radius at the head.",
            ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGuiComponents::SliderFloatField("Final radius", &bfr, 0.5f, 100.0f, "%.1f",
            "Radius at the head (m).", ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGuiComponents::SliderFloatField("Bullet speed", &bs, 1.0f, 1000.0f, "%.0f",
            "Flight speed (m/s).", ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGuiComponents::SliderFloatField("Bullet length", &bl, 10.0f, 2000.0f, "%.0f",
            "Path length (m).", ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGuiComponents::SliderFloatField("Bullet angle", &ba, 0.0f, 360.0f, "%.0f",
            "Flight direction in the XZ plane, 0 = +X.");
        {
            ImGuiComponents::FieldLabel("Loop duration",
                "Auto bullet: one shot per loop duration. Manually fired bullets are one-shot.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Loop duration##bullet");
            if (ImGui::SliderFloat("##v", &bld, 1.0f, 60.0f, "%.0f")) {
                bChanged = true;
            }
            ImGui::PopID();
            ImGuiComponents::TooltipOnHover("Auto bullet: one shot per loop duration. Manually fired bullets are one-shot.");
        }
        if (bChanged) {
            sdfRenderer->setBulletDefaults(br, bfr, bs, bl, ba, bld);
        }
        float ts = cfg.bullet.tunnelStrength;
        float tf = cfg.bullet.tunnelFalloff;
        bool tChanged = false;
        tChanged |= ImGuiComponents::SliderFloatField("Tunnel strength", &ts, 0.0f, 1.0f, "%.2f",
            "How strongly the tunnel core thins the smoke.");
        tChanged |= ImGuiComponents::SliderFloatField("Tunnel falloff", &tf, 0.5f, 100.0f, "%.1f",
            "Tunnel edge softness (m).", ImGuiSliderFlags_Logarithmic);
        if (tChanged) {
            sdfRenderer->setSmokeTunnel(ts, tf);
        }
        bool af = cfg.bullet.autoFire;
        if (ImGuiComponents::CheckboxField("Auto-fire each loop", &af,
                "Fire the template bullet automatically every loop.")) {
            sdfRenderer->setAutoFire(af);
        }
        if (ImGui::Button("Fire bullet")) {
            sdfRenderer->fireBullet(cfg.bullet.angleDeg);
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear bullets")) {
            sdfRenderer->clearBullets();
        }
        ImGui::Text("bullets in flight: %u", sdfRenderer->bulletSlotsUsed());
    });

    // 4: Pressure.
    sections.emplace_back([this]() {
        ImGui::Text("Pressure");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        float a = cfg.bullet.pressureRadius;
        float b = cfg.bullet.pressureStrength;
        float c = cfg.bullet.pressureWaveSpeed;
        float d = cfg.bullet.pressureWaveFreq;
        float e = cfg.bullet.pressureWaveFalloff;
        bool changed = false;
        {
            ImGuiComponents::FieldLabel("Radius", "Pressure field radius (m).");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Radius##pressure");
            if (ImGui::SliderFloat("##v", &a, 0.5f, 150.0f, "%.1f")) changed = true;
            ImGui::PopID();
        }
        {
            ImGuiComponents::FieldLabel("Strength", "Pressure displacement strength.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Strength##pressure");
            if (ImGui::SliderFloat("##v", &b, 0.0f, 30.0f, "%.1f")) changed = true;
            ImGui::PopID();
        }
        changed |= ImGuiComponents::SliderFloatField("Wave speed", &c, 0.0f, 300.0f, "%.0f",
            "Shock wave speed (m/s).");
        changed |= ImGuiComponents::SliderFloatField("Wave freq", &d, 0.01f, 3.0f, "%.2f",
            "Shock wave frequency (1/m).");
        changed |= ImGuiComponents::SliderFloatField("Wave falloff", &e, 0.001f, 1.0f, "%.3f",
            "Shock wave falloff (1/m).");
        if (changed) {
            sdfRenderer->setSmokePressure(a, b, c, d, e);
        }
    });

    // 5: Wake.
    sections.emplace_back([this]() {
        ImGui::Text("Wake");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        float a = cfg.bullet.wakeStrength;
        float b = cfg.bullet.wakeRadius;
        float c = cfg.bullet.wakeExpansion;
        float d = cfg.bullet.wakeLength;
        float e = cfg.bullet.wakeDissipation;
        bool changed = false;
        {
            ImGuiComponents::FieldLabel("Strength", "Wake strength.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Strength##wake");
            if (ImGui::SliderFloat("##v", &a, 0.0f, 2.0f, "%.2f")) changed = true;
            ImGui::PopID();
        }
        {
            ImGuiComponents::FieldLabel("Radius", "Wake radius (m).");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Radius##wake");
            if (ImGui::SliderFloat("##v", &b, 0.5f, 100.0f, "%.1f")) changed = true;
            ImGui::PopID();
        }
        changed |= ImGuiComponents::SliderFloatField("Expansion", &c, 0.0f, 1.0f, "%.2f",
            "Wake expansion rate.");
        changed |= ImGuiComponents::SliderFloatField("Length", &d, 1.0f, 500.0f, "%.0f",
            "Wake length (m).");
        {
            ImGuiComponents::FieldLabel("Dissipation", "Dissipation doubles as the tunnel refill rate.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Dissipation##wake");
            if (ImGui::SliderFloat("##v", &e, 0.05f, 3.0f, "%.2f")) changed = true;
            ImGui::PopID();
            ImGuiComponents::TooltipOnHover("Dissipation doubles as the tunnel refill rate.");
        }
        if (changed) {
            sdfRenderer->setSmokeWake(a, b, c, d, e);
        }
    });

    // 6: Turbulence.
    sections.emplace_back([this]() {
        ImGui::Text("Turbulence");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        float a = cfg.bullet.turbScale;
        float b = cfg.bullet.turbStrength;
        float c = cfg.bullet.turbSpeed;
        bool changed = false;
        changed |= ImGuiComponents::SliderFloatField("Scale", &a, 0.001f, 1.0f, "%.3f",
            "Turbulence scale (1/m).", ImGuiSliderFlags_Logarithmic);
        {
            ImGuiComponents::FieldLabel("Strength", "Turbulence strength.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Strength##turbulence");
            if (ImGui::SliderFloat("##v", &b, 0.0f, 10.0f, "%.2f")) changed = true;
            ImGui::PopID();
        }
        changed |= ImGuiComponents::SliderFloatField("Speed", &c, 0.0f, 10.0f, "%.2f",
            "Turbulence speed (1/s).");
        if (changed) {
            sdfRenderer->setSmokeTurbulence(a, b, c);
        }
    });

    // 7: Gold tracer.
    sections.emplace_back([this]() {
        ImGui::Text("Gold");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        glm::vec3 deep = cfg.bullet.goldDeep;
        glm::vec3 bright = cfg.bullet.goldBright;
        float sp = cfg.bullet.goldSpecPower;
        float ss = cfg.bullet.goldSpecStrength;
        float fb = cfg.bullet.goldFresnelBoost;
        float wf = cfg.bullet.goldWarmFloor;
        float ps = cfg.bullet.goldPatternScale;
        float nd = cfg.bullet.goldNormalDistort;
        bool changed = false;
        changed |= ImGuiComponents::ColorEdit3Field("Deep gold", &deep[0],
            "Deep gold tint.");
        changed |= ImGuiComponents::ColorEdit3Field("Bright gold", &bright[0],
            "Bright champagne gold tint.");
        changed |= ImGuiComponents::SliderFloatField("Spec power", &sp, 1.0f, 256.0f, "%.0f",
            "Specular power.", ImGuiSliderFlags_Logarithmic);
        changed |= ImGuiComponents::SliderFloatField("Spec strength", &ss, 0.0f, 10.0f, "%.2f",
            "Specular strength.");
        changed |= ImGuiComponents::SliderFloatField("Fresnel boost", &fb, 0.0f, 4.0f, "%.2f",
            "Fresnel boost.");
        changed |= ImGuiComponents::SliderFloatField("Warm floor", &wf, 0.0f, 1.0f, "%.2f",
            "Warm floor tint.");
        changed |= ImGuiComponents::SliderFloatField("Pattern scale", &ps, 0.05f, 8.0f, "%.2f",
            "Noise pattern scale.", ImGuiSliderFlags_Logarithmic);
        changed |= ImGuiComponents::SliderFloatField("Normal distort", &nd, 0.0f, 2.0f, "%.2f",
            "Normal distortion.");
        if (changed) {
            sdfRenderer->setBulletGold(deep, bright, sp, ss, fb, wf, ps, nd);
        }
    });

    // 8: Rendering.
    sections.emplace_back([this]() {
        ImGui::Text("Rendering");
        ImGuiComponents::ColSeparator();
        SdfEffectConfig& cfg = sdfRenderer->config();
        int ss = cfg.smoke.shadowSamples;
        float sh = cfg.smoke.shadowStrength;
        bool shadowChanged = false;
        shadowChanged |= ImGuiComponents::SliderIntField("Shadow samples", &ss, 1, 8, "%d",
            "Sun self-shadow march samples.");
        shadowChanged |= ImGuiComponents::SliderFloatField("Shadow strength", &sh, 0.0f, 1.0f, "%.2f",
            "Self-shadow strength.");
        if (shadowChanged) {
            sdfRenderer->setSmokeShadow(ss, sh);
        }
        float dsv = cfg.smoke.densityScale;
        if (ImGuiComponents::SliderFloatField("Density scale", &dsv, 0.0f, 3.0f, "%.2f",
                "Live density multiplier (no rebuild).")) {
            sdfRenderer->setSmokeDensityScale(dsv);
        }
        float ab = cfg.smoke.absorption;
        if (ImGuiComponents::SliderFloatField("Absorption", &ab, 0.0f, 3.0f, "%.2f",
                "Absorption coefficient (rebuilds scene).")) {
            sdfRenderer->setSmokeAbsorption(ab);
        }
        float sc = cfg.smoke.scattering;
        if (ImGuiComponents::SliderFloatField("Scattering", &sc, 0.0f, 8.0f, "%.2f",
                "Sun scatter brightness (default 4 = reference).")) {
            sdfRenderer->setSmokeScattering(sc);
        }
        glm::vec3 scc = cfg.smoke.smokeColor;
        if (ImGuiComponents::ColorEdit3Field("Smoke color", &scc[0],
                "Lit albedo tint of the smoke (shadowed end scales with it).")) {
            sdfRenderer->setSmokeColor(scc);
        }
        int dbg = static_cast<int>(cfg.smoke.debugView);
        const char* dbgItems[] = {"Normal smoke", "1 base SDF", "2 density", "3 bullet SDF",
                                  "4 tunnel", "5 compression", "6 wave", "7 air SVF (m/s)",
                                  "8 wake", "9 final density", "10 ray steps", "11 heat"};
        {
            ImGuiComponents::FieldLabel("Debug view", "Smoke debug visualization.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Debug view##smoke");
            if (ImGui::Combo("##v", &dbg, dbgItems, 12)) {
                sdfRenderer->setSmokeDebug(static_cast<uint32_t>(dbg));
            }
            ImGui::PopID();
        }
    });

    // 9: Wind debug (visualizes the shared wind field; simulated in the
    // wind widget, rendered here).
    sections.emplace_back([this]() {
        ImGui::Text("Wind Debug");
        ImGuiComponents::ColSeparator();
        if (!vegetationRenderer) {
            ImGui::TextWrapped("No vegetation renderer bound.");
            return;
        }
        WindFieldSettings& wfield = vegetationRenderer->getWindFieldSettings();
        int wdbg = wfield.windDebugMode;
        const char* wdbgItems[] = {"Off", "Heat map", "Velocity surface"};
        {
            ImGuiComponents::FieldLabel("Wind debug", "Raymarched overlay of the final wind field.");
            ImGui::SetNextItemWidth(kRaymarchColWidth);
            ImGui::PushID("Wind debug##raymarch");
            if (ImGui::Combo("##v", &wdbg, wdbgItems, 3)) {
                wfield.windDebugMode = wdbg < 0 ? 0 : (wdbg > 2 ? 2 : wdbg);
            }
            ImGui::PopID();
            ImGuiComponents::TooltipOnHover("Off = normal composite. Heat map = air-pressure heat + direction. "
                                            "Velocity surface = SDF isosurface where wind speed equals the iso below.");
        }
        float wiso = wfield.windDebugIso;
        if (ImGuiComponents::SliderFloatField("Surface speed", &wiso, 0.0f, 60.0f, "%.1f",
                "Isosurface speed in m/s (velocity-surface mode only).")) {
            wfield.windDebugIso = wiso < 0.0f ? 0.0f : wiso;
        }
    });

    // ---- Packing: same best-fit columns as the Settings menu (fixed 256px
    // ---- columns, overflow scrolls horizontally).
    const int n = static_cast<int>(sections.size());
    static std::vector<float> cachedH;
    if (static_cast<int>(cachedH.size()) != n) {
        cachedH.assign(n, 170.0f);
        cachedH[0] = 300.0f; // Raymarcher
        cachedH[1] = 480.0f; // Fire
        cachedH[2] = 420.0f; // Shape (translation, scale, rotation)
        cachedH[3] = 520.0f; // Smoke
        cachedH[4] = 430.0f; // Bullet
        cachedH[5] = 280.0f; // Pressure
        cachedH[6] = 300.0f; // Wake
        cachedH[7] = 200.0f; // Turbulence
        cachedH[8] = 430.0f; // Gold
        cachedH[9] = 330.0f; // Rendering
        cachedH[10] = 200.0f; // Wind Debug
    }

    float availW = ImGui::GetContentRegionAvail().x;
    float availH = ImGui::GetContentRegionAvail().y;
    if (availW < 50.0f) availW = 300.0f;
    if (availH < 50.0f) availH = 500.0f;
    const float flowH = availH - 4.0f;

    constexpr float kSectionGap = 8.0f;
    std::vector<int> colOf(n, 0);
    std::vector<float> colH(1, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float h = cachedH[i] > 1.0f ? cachedH[i] : 170.0f;
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

    constexpr float colWidth = kRaymarchColWidth;
    const float spacingX = ImGui::GetStyle().ItemSpacing.x;

    for (int c = 0; c < nCols; ++c) {
        if (c > 0) ImGui::SameLine(0.0f, spacingX);
        ImGui::BeginGroup();
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
