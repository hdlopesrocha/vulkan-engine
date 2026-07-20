#pragma once

#include "Widget.hpp"
#include "../events/ControllerManager.hpp"
#include <imgui.h>

class Brush3dManager;

class ControllerParametersWidget : public Widget {
public:
    ControllerParametersWidget(ControllerManager* cm, Brush3dManager* brushManager);
    void render() override;

private:
    ControllerManager* cm;
    Brush3dManager* brushManager;
};
