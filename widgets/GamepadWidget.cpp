#include "GamepadWidget.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include "../events/ControllerParameters.hpp"
#include <imgui.h>

static void drawPageIcon(ControllerParameters::pageType p) {
    const char* label = "?";
    ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    switch (p) {
        case ControllerParameters::CAMERA: label = "CAM"; col = ImVec4(0.2f,0.5f,0.9f,1.0f); break;
        case ControllerParameters::BRUSH_POSITION: label = "POS"; col = ImVec4(0.2f,0.9f,0.3f,1.0f); break;
        case ControllerParameters::BRUSH_SCALE: label = "SCL"; col = ImVec4(0.95f,0.6f,0.1f,1.0f); break;
        case ControllerParameters::BRUSH_ROTATION: label = "ROT"; col = ImVec4(0.7f,0.3f,0.9f,1.0f); break;
        case ControllerParameters::BRUSH_PROPERTIES: label = "PRP"; col = ImVec4(0.6f,0.6f,0.6f,1.0f); break;
    }

    // Reserve space and draw a small colored circle then the label
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(36, 0));
    ImVec2 center = ImVec2(pos.x + 10.0f, pos.y + 10.0f);
    dl->AddCircleFilled(center, 8.0f, ImGui::ColorConvertFloat4ToU32(col));
    dl->AddText(ImVec2(pos.x + 22.0f, pos.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)), label);
}

static const char* buttonName(int b) {
    switch (b) {
        case GLFW_GAMEPAD_BUTTON_A: return "A";
        case GLFW_GAMEPAD_BUTTON_B: return "B";
        case GLFW_GAMEPAD_BUTTON_X: return "X";
        case GLFW_GAMEPAD_BUTTON_Y: return "Y";
        case GLFW_GAMEPAD_BUTTON_LEFT_BUMPER: return "LB";
        case GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER: return "RB";
        case GLFW_GAMEPAD_BUTTON_BACK: return "BACK";
        case GLFW_GAMEPAD_BUTTON_GUIDE: return "GUIDE";
        case GLFW_GAMEPAD_BUTTON_START: return "START";
        case GLFW_GAMEPAD_BUTTON_LEFT_THUMB: return "L_THUMB";
        case GLFW_GAMEPAD_BUTTON_RIGHT_THUMB: return "R_THUMB";
        case GLFW_GAMEPAD_BUTTON_DPAD_UP: return "DUP";
        case GLFW_GAMEPAD_BUTTON_DPAD_RIGHT: return "DR";
        case GLFW_GAMEPAD_BUTTON_DPAD_DOWN: return "DD";
        case GLFW_GAMEPAD_BUTTON_DPAD_LEFT: return "DL";
        default: return "?";
    }
}

GamepadWidget::GamepadWidget(ControllerParameters* params) : Widget("Gamepad", u8"\uf11b"), ctrlParams(params) {}

void GamepadWidget::render() {
    if (!ImGui::Begin(displayTitle().c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    // Build a list of joysticks with their presence/gamepad status
    std::vector<std::string> labels;
    labels.reserve(GLFW_JOYSTICK_LAST - GLFW_JOYSTICK_1 + 1);
    int defaultIndex = 0;
    for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; ++jid) {
        const char* name = glfwGetJoystickName(jid);
        bool present = glfwJoystickPresent(jid);
        bool isGp = glfwJoystickIsGamepad(jid);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "J%d: %s %s", jid - GLFW_JOYSTICK_1 + 1,
                      present ? (name ? name : "(unknown)") : "(absent)",
                      isGp ? "(gamepad)" : "");
        labels.emplace_back(buf);
        if (jid == selectedJoystick) defaultIndex = jid - GLFW_JOYSTICK_1;
    }

    // Prepare array of c_str for ImGui combo
    std::vector<const char*> items;
    items.reserve(labels.size());
    for (auto &s : labels) items.push_back(s.c_str());

    int idx = selectedJoystick - GLFW_JOYSTICK_1;
    if (ImGui::Combo("Joystick", &idx, items.data(), (int)items.size())) {
        selectedJoystick = GLFW_JOYSTICK_1 + idx;
    }

    bool present = glfwJoystickPresent(selectedJoystick);
    ImGui::Text("Present: %s", present ? "Yes" : "No");
    if (!present) {
        ImGui::End();
        return;
    }

    bool isGamepad = glfwJoystickIsGamepad(selectedJoystick);
    ImGui::Text("Recognized as gamepad: %s", isGamepad ? "Yes" : "No");

    // If it is a mapped gamepad, show axes/buttons
    if (isGamepad) {
        GLFWgamepadstate state;
        if (glfwGetGamepadState(selectedJoystick, &state)) {
            ImGui::Separator();
            ImGui::Text("Axes");
            auto showAxis = [&](const char* label, float v) {
                float raw = v;
                // Map [-1,1] to [0,1] for progress bar
                float mapped = (raw + 1.0f) * 0.5f;
                ImGui::Text("%s: %.2f", label, raw);
                ImGui::SameLine(200);
                ImGui::ProgressBar(mapped, ImVec2(0.0f, 0.0f));
            };

            showAxis("Left X", state.axes[GLFW_GAMEPAD_AXIS_LEFT_X]);
            showAxis("Left Y", state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]);
            showAxis("Right X", state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X]);
            showAxis("Right Y", state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]);
            showAxis("L Trigger", state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER]);
            showAxis("R Trigger", state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER]);

            ImGui::Separator();
            ImGui::Text("Buttons");
            ImGui::Columns(4, "gp_buttons_cols", false);
            for (int b = GLFW_GAMEPAD_BUTTON_A; b <= GLFW_GAMEPAD_BUTTON_LAST; ++b) {
                bool pressed = (state.buttons[b] == GLFW_PRESS);
                const char* name = buttonName(b);
                if (pressed) ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "%s", name);
                else ImGui::Text("%s", name);
                ImGui::NextColumn();
            }
            ImGui::Columns(1);
        } else {
            ImGui::Text("Failed to read gamepad state");
        }
    } else {
        ImGui::TextWrapped("This joystick is not recognized as a mapped gamepad. If you expect it to be a gamepad, ensure mappings are available (glfwUpdateGamepadMappings) or connect a supported controller.");
    }

    // Show controller page indicator after the gamepad state (axes/buttons)
    if (ctrlParams) {
        ImGui::Separator();
        drawPageIcon(ctrlParams->currentPage);
    }

    ImGui::End();
}
