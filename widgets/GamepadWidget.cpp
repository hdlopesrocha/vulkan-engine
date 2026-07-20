#include "GamepadWidget.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include "../events/ControllerManager.hpp"
#include "../events/ControllerContext.hpp"
#include "../events/NunchukPublisher.hpp"
#include <wiiuse.h>
#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

static void drawPageIcon(const ControllerContext& ctx) {
    const char* label = "?";
    ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    switch (ctx.activeCategory()) {
        case PageCategory::CAMERA:
            label = "CAM"; col = ImVec4(0.2f, 0.5f, 0.9f, 1.0f); break;
        case PageCategory::BRUSH:
            label = "BRU"; col = ImVec4(0.2f, 0.9f, 0.3f, 1.0f); break;
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

GamepadWidget::GamepadWidget(ControllerManager* cm, NunchukPublisher* nunchuk)
    : Widget("Gamepad", u8"\uf11b"), ctrlManager(cm), nunchukPublisher(nunchuk) {}

void GamepadWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // ── Standard Gamepad section ──
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

    bool isGamepad = glfwJoystickIsGamepad(selectedJoystick);
    ImGui::Text("Recognized as gamepad: %s", isGamepad ? "Yes" : "No");

    // If it is a mapped gamepad, show axes/buttons
    if (present && isGamepad) {
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
    if (ctrlManager) {
        ImGui::Separator();
        drawPageIcon(ctrlManager->gamepadContext);
        ImGui::Text("Page: %s > %s", ctrlManager->gamepadContext.activePageName().c_str(),
                    ctrlManager->gamepadContext.activeSubpageName().c_str());
    }

    // ── Wiimote section ──
    ImGui::Separator();
    ImGui::TextUnformatted("Wiimote (wiiuse)");

    if (!nunchukPublisher) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Wiimote publisher not available");
        return;
    }

    WiimoteState nc = nunchukPublisher->getState();

    if (!nc.connected) {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.2f, 1.0f), "Disconnected");
        if (nunchukPublisher->isAutoConnecting()) {
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.2f, 1.0f), "Auto-connecting... (press 1+2 on Wiimote)");
        }
        if (ImGui::Button("Stop Auto-Connect")) {
            nunchukPublisher->disconnect();
        }
        return;
    }

    // Wiimote connected - show all inputs/sensors
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Connected");
    ImGui::SameLine();
    if (ImGui::SmallButton("Disconnect")) {
        nunchukPublisher->disconnect();
        return;
    }

    // LEDs + battery
    ImGui::Text("LEDs:");
    ImGui::SameLine(60);
    auto ledTxt = [&](const char* name, bool on) {
        if (on) ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "%s", name);
        else ImGui::Text("%s", name);
        ImGui::SameLine();
    };
    ledTxt("1", nc.led1); ledTxt("2", nc.led2); ledTxt("3", nc.led3); ledTxt("4", nc.led4);
    ImGui::NewLine();
    ImGui::Text("Battery:");
    ImGui::SameLine(80);
    ImGui::Text("%.0f%%", nc.batteryLevel * 100.0f);

    // Wiimote buttons
    ImGui::Separator();
    ImGui::Text("Wiimote Buttons");
    auto wbtn = [&](const char* label, uint16_t mask) {
        if (nc.buttons & mask) ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "%s", label);
        else ImGui::Text("%s", label);
        ImGui::SameLine();
    };
    wbtn("A", WIIMOTE_BUTTON_A); wbtn("B", WIIMOTE_BUTTON_B);
    wbtn("1", WIIMOTE_BUTTON_ONE); wbtn("2", WIIMOTE_BUTTON_TWO);
    wbtn("+", WIIMOTE_BUTTON_PLUS); wbtn("-", WIIMOTE_BUTTON_MINUS);
    wbtn("Home", WIIMOTE_BUTTON_HOME);
    wbtn("Up", WIIMOTE_BUTTON_UP); wbtn("Down", WIIMOTE_BUTTON_DOWN);
    wbtn("Left", WIIMOTE_BUTTON_LEFT); wbtn("Right", WIIMOTE_BUTTON_RIGHT);
    ImGui::NewLine();

    // Wiimote accelerometer + orientation
    ImGui::Text("Accel: X=%d  Y=%d  Z=%d", nc.accelX, nc.accelY, nc.accelZ);
    ImGui::Text("Orient: roll=%.1f  pitch=%.1f  yaw=%.1f", nc.roll, nc.pitch, nc.yaw);
    ImGui::Text("G-Force: x=%.2f  y=%.2f  z=%.2f", nc.gforceX, nc.gforceY, nc.gforceZ);

    // IR camera
    ImGui::Separator();
    ImGui::Text("IR Camera: %s", nc.irEnabled ? "on" : "off");
    ImGui::Text("Dots: %d   Pos: (%d, %d)   Distance: %.1f", nc.irNumDots, nc.irX, nc.irY, nc.irDistance);

    // Expansion / Nunchuk
    ImGui::Separator();
    const char* expName = "None";
    switch (nc.expansionType) {
        case EXP_NUNCHUK: expName = "Nunchuk"; break;
        case EXP_MOTION_PLUS_NUNCHUK: expName = "Motion+ + Nunchuk"; break;
        case EXP_MOTION_PLUS: expName = "Motion+"; break;
        case EXP_CLASSIC: expName = "Classic"; break;
        case EXP_MOTION_PLUS_CLASSIC: expName = "Motion+ + Classic"; break;
        case EXP_GUITAR_HERO_3: expName = "Guitar Hero 3"; break;
        case EXP_WII_BOARD: expName = "Wii Board"; break;
        case EXP_TATACON: expName = "Taiko Drum"; break;
    }
    ImGui::Text("Expansion: %s", expName);

    if (nc.expansionConnected && (nc.expansionType == EXP_NUNCHUK || nc.expansionType == EXP_MOTION_PLUS_NUNCHUK)) {
        // Nunchuk joystick
        ImGui::Text("Joystick: (%.2f, %.2f)  ang=%.0f  mag=%.2f", nc.joystickX, nc.joystickY, nc.joystickAngle, nc.joystickMag);
        ImGui::SameLine(260);
        ImGui::ProgressBar((nc.joystickX + 1.0f) * 0.5f, ImVec2(50.0f, 0.0f), "");
        ImGui::SameLine(320);
        ImGui::ProgressBar((nc.joystickY + 1.0f) * 0.5f, ImVec2(50.0f, 0.0f), "");

        ImGui::Text("Buttons:");
        ImGui::SameLine(80);
        if (nc.buttonC) ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "C");
        else ImGui::Text("C");
        ImGui::SameLine(120);
        if (nc.buttonZ) ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.2f, 1.0f), "Z");
        else ImGui::Text("Z");

        ImGui::Text("Accel: X=%d  Y=%d  Z=%d", nc.nunchukAccelX, nc.nunchukAccelY, nc.nunchukAccelZ);
        ImGui::Text("Orient: roll=%.1f  pitch=%.1f  yaw=%.1f", nc.nunchukRoll, nc.nunchukPitch, nc.nunchukYaw);
        ImGui::Text("G-Force: x=%.2f  y=%.2f  z=%.2f", nc.nunchukGforceX, nc.nunchukGforceY, nc.nunchukGforceZ);
    } else if (nc.expansionType == EXP_NONE) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No expansion (Wiimote only)");
    }
}
