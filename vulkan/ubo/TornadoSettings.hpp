#pragma once

#include <glm/glm.hpp>

// Widget-facing settings (edited by the wind widget; packed to the wire
// struct by VegetationRenderer::updateWindFieldUBO). Ambient sliders are NOT
// duplicated here — ambient is mirrored from WindSettings at pack time.
struct TornadoSettings {
    glm::vec2 baseXZ = glm::vec2(0.0f, 0.0f);
    float groundY = 0.0f; // sea level default; see grounding note in WindField.hpp
    float radius = 25.0f;
    float height = 120.0f;
    float strength = 30.0f;
    float direction = 1.0f; // -1..1
    float phase = 0.0f;
    float deltaTime = 1.0f;
    glm::vec2 driftVelocity = glm::vec2(0.0f, 0.0f);
    float wanderRadius = 0.0f;
    float wanderSpeed = 0.0f;
    float swingAmplitude = 0.0f; // 0 = static direction
    float swingFrequency = 0.0f;
    bool active = false;
};
