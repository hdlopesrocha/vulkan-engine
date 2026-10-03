#pragma once

#include <glm/glm.hpp>

// Separate UBO dedicated to sky parameters (binding 6).
// The cloud block extends the same UBO so the sky pass (which already binds
// set 0 / binding 6) needs no descriptor-layout change: only the buffer size
// grows. Terrain/water shaders also read binding 6 for cloud shadows.
struct SkyUniform {
    glm::vec4 skyHorizon; // rgb = horizon color, a = unused
    glm::vec4 skyZenith;  // rgb = zenith color, a = unused
    glm::vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = skyMode
    glm::vec4 nightHorizon; // rgb = night horizon color
    glm::vec4 nightZenith;  // rgb = night zenith color
    glm::vec4 nightParams;  // x = night intensity, y = starIntensity, z/w unused
    // --- Volumetric clouds (see widgets/CloudSettings.hpp) ---
    glm::vec4 cloudToggles; // x = enabled, y = lowOn, z = midOn, w = highOn
    glm::vec4 cloudGlobal;  // x = densityScale, y = windSpeed, z = windAngleRad, w = detailStrength
    glm::vec4 cloudTime;    // x = time, y = shadowStrength, z = raymarchSteps, w = lightSteps
    glm::vec4 cloudLow;     // x = coverage, y = density, z = scale, w = windSpeedMul
    glm::vec4 cloudLowGeom; // x = baseHeight, y = thickness, zw unused
    glm::vec4 cloudMid;     // x = coverage, y = density, z = scale, w = windSpeedMul
    glm::vec4 cloudMidGeom; // x = baseHeight, y = thickness, zw unused
    glm::vec4 cloudHigh;    // x = coverage, y = density, z = scale, w = windSpeedMul
    glm::vec4 cloudHighGeom;// x = baseHeight, y = thickness, zw unused
    glm::vec4 cloudLight;   // x = silverLining, y = ambientBoost, z = sunForwardG, w = exposure
    glm::vec4 cloudAnim;    // x = timeScale, yzw unused (time already advanced on CPU)
};
