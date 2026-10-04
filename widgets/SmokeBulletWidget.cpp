#include "SmokeBulletWidget.hpp"

#include <imgui.h>
#include "../math/Camera.hpp"
#include "components/ImGuiHelpers.hpp"

SmokeBulletWidget::SmokeBulletWidget(SdfRenderer* sdf, Camera* cam)
    : Widget("Smoke Bomb"), sdfRenderer(sdf), camera(cam) {
    isOpen = false;
}

void SmokeBulletWidget::render() {
    if (!sdfRenderer) return;
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    ImGui::TextWrapped("Procedural smoke grenade on the generic SDF renderer: "
                       "growing sphere, layered noise, bullet tunnel + pressure "
                       "wave + turbulent wake with refill. Animation loops every "
                       "loop duration.");
    ImGui::Separator();

    // All tuning below binds per-frame locals read from the shared config
    // (SdfRenderer owns the only stored copy); edits commit through the
    // renderer setters, which clamp and route uploads/rebuilds.
    SdfEffectConfig& cfg = sdfRenderer->config();

    if (ImGui::CollapsingHeader("Smoke", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool enabled = cfg.smoke.enabled;
        if (ImGui::Checkbox("Enable smoke", &enabled)) {
            sdfRenderer->setSmokeEnabled(enabled);
        }
        if (camera && ImGui::Button("Go to smoke (center camera)")) {
            camera->translate(cfg.smoke.pos - camera->getPosition());
        }
        ImGuiHelpers::SetTooltipIfHovered("Teleport the camera to the smoke center.");
        glm::vec3 pos = cfg.smoke.pos;
        if (ImGui::DragFloat3("Position", &pos[0], 1.0f)) {
            sdfRenderer->setSmokePosition(pos);
        }
        float v = cfg.smoke.radius;
        if (ImGui::SliderFloat("Max radius", &v, 8.0f, 1024.0f, "%.0f",
                               ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setSmokeRadius(v);
        }
        v = cfg.smoke.growthDuration;
        if (ImGui::SliderFloat("Growth time", &v, 0.5f, 60.0f, "%.1f")) {
            sdfRenderer->setSmokeGrowthDuration(v);
        }
        v = cfg.smoke.loopDuration;
        if (ImGui::SliderFloat("Loop duration", &v, 2.0f, 120.0f, "%.0f")) {
            sdfRenderer->setSmokeLoopDuration(v);
        }
        ImGuiHelpers::SetTooltipIfHovered("Smoke animation repeats every loop duration.");
        v = cfg.smoke.dissipation;
        if (ImGui::SliderFloat("Dissipation", &v, 0.0f, 2.0f, "%.2f")) {
            sdfRenderer->setSmokeDissipation(v);
        }
        v = cfg.smoke.density;
        if (ImGui::SliderFloat("Density", &v, 0.05f, 2.0f, "%.2f")) {
            sdfRenderer->setSmokeDensity(v);
        }
        v = cfg.smoke.noiseScale;
        if (ImGui::SliderFloat("Noise scale", &v, 0.001f, 0.2f, "%.4f",
                               ImGuiSliderFlags_Logarithmic)) {
            sdfRenderer->setSmokeNoiseScale(v);
        }
        v = cfg.smoke.noiseStrength;
        if (ImGui::SliderFloat("Noise strength", &v, 0.0f, 2.0f, "%.2f")) {
            sdfRenderer->setSmokeNoiseStrength(v);
        }
        v = cfg.smoke.noiseWarp;
        if (ImGui::SliderFloat("Noise warp", &v, 0.0f, 2.0f, "%.2f")) {
            sdfRenderer->setSmokeNoiseWarp(v);
        }
        v = cfg.smoke.windSpeed;
        float wa = cfg.smoke.windAngleDeg;
        bool windChanged = false;
        windChanged |= ImGui::SliderFloat("Wind speed", &v, 0.0f, 60.0f, "%.1f");
        windChanged |= ImGui::SliderFloat("Wind angle", &wa, 0.0f, 360.0f, "%.0f");
        if (windChanged) {
            sdfRenderer->setSmokeWind(v, wa);
        }
    }

    if (ImGui::CollapsingHeader("Bullet", ImGuiTreeNodeFlags_DefaultOpen)) {
        float br = cfg.bullet.radius;
        float bfr = cfg.bullet.finalRadius;
        float bs = cfg.bullet.speed;
        float bl = cfg.bullet.length;
        float ba = cfg.bullet.angleDeg;
        float bld = cfg.bullet.loopDuration;
        bool bChanged = false;
        bChanged |= ImGui::SliderFloat("Bullet radius", &br, 0.5f, 100.0f, "%.1f",
                                       ImGuiSliderFlags_Logarithmic);
        ImGuiHelpers::SetTooltipIfHovered("Initial radius at launch (m); the tunnel tapers "
                                          "to the final radius at the head.");
        bChanged |= ImGui::SliderFloat("Final radius", &bfr, 0.5f, 256.0f, "%.1f",
                                       ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGui::SliderFloat("Bullet speed", &bs, 1.0f, 1000.0f, "%.0f",
                                       ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGui::SliderFloat("Bullet length", &bl, 10.0f, 2000.0f, "%.0f",
                                       ImGuiSliderFlags_Logarithmic);
        bChanged |= ImGui::SliderFloat("Bullet angle", &ba, 0.0f, 360.0f, "%.0f");
        bChanged |= ImGui::SliderFloat("Loop duration", &bld, 1.0f, 60.0f, "%.0f");
        if (bChanged) {
            sdfRenderer->setBulletDefaults(br, bfr, bs, bl, ba, bld);
        }
        ImGuiHelpers::SetTooltipIfHovered("Auto bullet: one shot per loop duration. Manually fired bullets are one-shot.");
        float ts = cfg.bullet.tunnelStrength;
        float tf = cfg.bullet.tunnelFalloff;
        bool tChanged = false;
        tChanged |= ImGui::SliderFloat("Tunnel strength", &ts, 0.0f, 1.0f, "%.2f");
        tChanged |= ImGui::SliderFloat("Tunnel falloff", &tf, 0.5f, 100.0f, "%.1f",
                                       ImGuiSliderFlags_Logarithmic);
        if (tChanged) {
            sdfRenderer->setSmokeTunnel(ts, tf);
        }
        bool af = cfg.bullet.autoFire;
        if (ImGui::Checkbox("Auto-fire each loop", &af)) {
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
    }

    if (ImGui::CollapsingHeader("Pressure")) {
        float a = cfg.bullet.pressureRadius;
        float b = cfg.bullet.pressureStrength;
        float c = cfg.bullet.pressureWaveSpeed;
        float d = cfg.bullet.pressureWaveFreq;
        float e = cfg.bullet.pressureWaveFalloff;
        bool changed = false;
        changed |= ImGui::SliderFloat("Radius", &a, 0.5f, 150.0f, "%.1f");
        changed |= ImGui::SliderFloat("Strength", &b, 0.0f, 30.0f, "%.1f");
        changed |= ImGui::SliderFloat("Wave speed", &c, 0.0f, 300.0f, "%.0f");
        changed |= ImGui::SliderFloat("Wave freq", &d, 0.01f, 3.0f, "%.2f");
        changed |= ImGui::SliderFloat("Wave falloff", &e, 0.001f, 1.0f, "%.3f");
        if (changed) {
            sdfRenderer->setSmokePressure(a, b, c, d, e);
        }
    }

    if (ImGui::CollapsingHeader("Wake")) {
        float a = cfg.bullet.wakeStrength;
        float b = cfg.bullet.wakeRadius;
        float c = cfg.bullet.wakeExpansion;
        float d = cfg.bullet.wakeLength;
        float e = cfg.bullet.wakeDissipation;
        bool changed = false;
        changed |= ImGui::SliderFloat("Strength", &a, 0.0f, 2.0f, "%.2f");
        changed |= ImGui::SliderFloat("Radius", &b, 0.5f, 100.0f, "%.1f");
        changed |= ImGui::SliderFloat("Expansion", &c, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Length", &d, 1.0f, 500.0f, "%.0f");
        changed |= ImGui::SliderFloat("Dissipation", &e, 0.05f, 3.0f, "%.2f");
        if (changed) {
            sdfRenderer->setSmokeWake(a, b, c, d, e);
        }
        ImGuiHelpers::SetTooltipIfHovered("Dissipation doubles as the tunnel refill rate.");
    }

    if (ImGui::CollapsingHeader("Turbulence")) {
        float a = cfg.bullet.turbScale;
        float b = cfg.bullet.turbStrength;
        float c = cfg.bullet.turbSpeed;
        bool changed = false;
        changed |= ImGui::SliderFloat("Scale", &a, 0.001f, 1.0f, "%.3f",
                                      ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Strength", &b, 0.0f, 10.0f, "%.2f");
        changed |= ImGui::SliderFloat("Speed", &c, 0.0f, 10.0f, "%.2f");
        if (changed) {
            sdfRenderer->setSmokeTurbulence(a, b, c);
        }
    }

    if (ImGui::CollapsingHeader("Gold")) {
        glm::vec3 deep = cfg.bullet.goldDeep;
        glm::vec3 bright = cfg.bullet.goldBright;
        float sp = cfg.bullet.goldSpecPower;
        float ss = cfg.bullet.goldSpecStrength;
        float fb = cfg.bullet.goldFresnelBoost;
        float wf = cfg.bullet.goldWarmFloor;
        float ps = cfg.bullet.goldPatternScale;
        float nd = cfg.bullet.goldNormalDistort;
        bool changed = false;
        changed |= ImGui::ColorEdit3("Deep gold", &deep[0]);
        changed |= ImGui::ColorEdit3("Bright gold", &bright[0]);
        changed |= ImGui::SliderFloat("Spec power", &sp, 1.0f, 256.0f, "%.0f",
                                      ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Spec strength", &ss, 0.0f, 10.0f, "%.2f");
        changed |= ImGui::SliderFloat("Fresnel boost", &fb, 0.0f, 4.0f, "%.2f");
        changed |= ImGui::SliderFloat("Warm floor", &wf, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Pattern scale", &ps, 0.05f, 8.0f, "%.2f",
                                      ImGuiSliderFlags_Logarithmic);
        changed |= ImGui::SliderFloat("Normal distort", &nd, 0.0f, 2.0f, "%.2f");
        if (changed) {
            sdfRenderer->setBulletGold(deep, bright, sp, ss, fb, wf, ps, nd);
        }
    }

    if (ImGui::CollapsingHeader("Rendering")) {
        int ss = cfg.smoke.shadowSamples;
        float sh = cfg.smoke.shadowStrength;
        bool shadowChanged = false;
        shadowChanged |= ImGui::SliderInt("Shadow samples", &ss, 1, 8);
        shadowChanged |= ImGui::SliderFloat("Shadow strength", &sh, 0.0f, 1.0f, "%.2f");
        if (shadowChanged) {
            sdfRenderer->setSmokeShadow(ss, sh);
        }
        float dsv = cfg.smoke.densityScale;
        if (ImGui::SliderFloat("Density scale", &dsv, 0.0f, 3.0f, "%.2f")) {
            sdfRenderer->setSmokeDensityScale(dsv);
        }
        float ab = cfg.smoke.absorption;
        if (ImGui::SliderFloat("Absorption", &ab, 0.0f, 3.0f, "%.2f")) {
            sdfRenderer->setSmokeAbsorption(ab);
        }
        float sc = cfg.smoke.scattering;
        if (ImGui::SliderFloat("Scattering", &sc, 0.0f, 2.0f, "%.2f")) {
            sdfRenderer->setSmokeScattering(sc);
        }
        int dbg = static_cast<int>(cfg.smoke.debugView);
        const char* dbgItems[] = {"Normal smoke", "1 base SDF", "2 density", "3 bullet SDF",
                                  "4 tunnel", "5 pressure", "6 wave", "7 turbulence",
                                  "8 wake", "9 final density", "10 ray steps"};
        if (ImGui::Combo("Debug view", &dbg, dbgItems, 11)) {
            sdfRenderer->setSmokeDebug(static_cast<uint32_t>(dbg));
        }
    }
}
