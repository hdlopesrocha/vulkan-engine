#pragma once

#include "Widget.hpp"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <string>

class ControllerManager;   // forward
class NunchukPublisher;     // forward

class GamepadWidget : public Widget {
public:
    GamepadWidget(ControllerManager* cm = nullptr, NunchukPublisher* nunchuk = nullptr);
    void render() override;

private:
    int selectedJoystick = GLFW_JOYSTICK_1;
    const float deadzone = 0.05f;
    ControllerManager* ctrlManager = nullptr;
    NunchukPublisher* nunchukPublisher = nullptr;
};
