#pragma once

#include "Widget.hpp"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <string>

class GamepadWidget : public Widget {
public:
    GamepadWidget();
    void render() override;

private:
    int selectedJoystick = GLFW_JOYSTICK_1;
    const float deadzone = 0.05f;
};
