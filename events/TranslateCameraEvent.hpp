#pragma once
#include "Event.hpp"
#include <glm/glm.hpp>
#include <string>

// Event: translate the camera by a delta vector (world or local-space as agreed by handler)
class TranslateCameraEvent : public Event {
public:
    TranslateCameraEvent() = default;
    TranslateCameraEvent(const glm::vec3 &d) : delta(d) {}
    explicit TranslateCameraEvent(float x, float y, float z) : delta(x,y,z) {}

    std::string name() const override { return "TranslateCameraEvent"; }

    glm::vec3 delta{0.0f, 0.0f, 0.0f};
};