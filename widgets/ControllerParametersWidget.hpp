#pragma once

#include "Widget.hpp"
#include "../events/ControllerParameters.hpp"
#include <imgui.h>

class Brush3dManager;

class ControllerParametersWidget : public Widget {
public:
    ControllerParametersWidget(ControllerParameters* params, Brush3dManager* brushManager);
    void render() override;

private:
    ControllerParameters* params;
    Brush3dManager* brushManager;
};
