#pragma once

#include "Widget.hpp"

class EventManager;

// Right-edge main-UI overlay with one-click graphics-quality presets. The
// widget only queues SetGraphicsQualityEvent; MyApp executes the matching
// GraphicsSettingsCommand (settings edits + per-layer water GPU upload), so
// the widget stays free of renderer state.
class GraphicsQualityWidget : public Widget {
public:
    explicit GraphicsQualityWidget(EventManager* eventManager);

    void render() override;

private:
    EventManager* eventManager = nullptr;
};
