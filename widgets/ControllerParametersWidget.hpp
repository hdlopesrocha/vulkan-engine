#pragma once

#include "Widget.hpp"
#include "../events/ControllerParameters.hpp"
#include <imgui.h>

class ControllerParametersWidget : public Widget {
public:
    ControllerParametersWidget(ControllerParameters* params);
    void render() override;

private:
    ControllerParameters* params;
};
