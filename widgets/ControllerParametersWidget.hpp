#pragma once

#include "Widget.hpp"
#include "ControllerParameters.hpp"
#include <imgui.h>

class ControllerParametersWidget : public Widget {
public:
    ControllerParametersWidget(ControllerParameters* params);
    void render() override;

private:
    ControllerParameters* params;
};
