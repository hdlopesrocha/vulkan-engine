#pragma once

#include "Widget.hpp"
#include <glm/glm.hpp>
#include <imgui.h>

enum class SkyMode {
    Gradient = 0,
    Grid = 1
};

#include "SkySettings.hpp"

class SkyWidget : public Widget {
private:
    // Reference to centralized settings owned externally (e.g., MyApp)
    SkySettings &settings;

public:
    explicit SkyWidget(SkySettings &settings);

    void render() override;

    // Forwarding accessors for compatibility
    glm::vec3 getHorizonColor() const { return settings.horizonColor; }
    glm::vec3 getZenithColor() const { return settings.zenithColor; }
    float getWarmth() const { return settings.warmth; }
    float getExponent() const { return settings.exponent; }
    float getSunFlare() const { return settings.sunFlare; }
    glm::vec3 getNightHorizon() const { return settings.nightHorizon; }
    glm::vec3 getNightZenith() const { return settings.nightZenith; }
    float getNightIntensity() const { return settings.nightIntensity; }
    float getStarIntensity() const { return settings.starIntensity; }
    SkyMode getSkyMode() const { return static_cast<SkyMode>(settings.mode);
    }
};
