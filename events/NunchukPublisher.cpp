#include "NunchukPublisher.hpp"
#include "EventManager.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "ControllerInput.hpp"
#include "../math/Camera.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include "RebuildBrushEvent.hpp"

#include <wiiuse.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>

static const int WIIMOTE_TIMEOUT_SEC = 5;

NunchukPublisher::NunchukPublisher() {
    wiiuse_set_output(LOGLEVEL_ERROR, stdout);
    wiiuse_set_output(LOGLEVEL_WARNING, stdout);
}

NunchukPublisher::~NunchukPublisher() {
    disconnect();
}

void NunchukPublisher::disconnect() {
    stopThread = true;
    autoConnecting = false;

    if (connectThread.joinable()) {
        connectThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        resetStateLocked();
    }

    if (wiimotes) {
        for (int i = 0; i < wiimoteCount; ++i) {
            if (wiimotes[i]) wiiuse_disconnect(wiimotes[i]);
        }
        wiiuse_cleanup(wiimotes, wiimoteCount);
        wiimotes = nullptr;
        wiimoteCount = 0;
    }
    connected = false;
}

void NunchukPublisher::connect() {
    if (autoConnecting.load()) return;
    stopThread = false;
    autoConnecting = true;
    connectThread = std::thread(&NunchukPublisher::autoConnectLoop, this);
}

void NunchukPublisher::resetStateLocked() {
    state = WiimoteState{};
}

void NunchukPublisher::autoConnectLoop() {
    while (!stopThread.load()) {
        fprintf(stdout, "[Wiimote] Scanning for Wiimote (press 1+2)...\n");

        wiimotes = wiiuse_init(1);
        if (!wiimotes) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        wiimoteCount = 1;

        int found = wiiuse_find(wiimotes, wiimoteCount, WIIMOTE_TIMEOUT_SEC);
        if (found <= 0) {
            fprintf(stderr, "[Wiimote] No Wiimote found\n");
            wiiuse_cleanup(wiimotes, wiimoteCount);
            wiimotes = nullptr;
            wiimoteCount = 0;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        if (wiiuse_connect(wiimotes, wiimoteCount) <= 0) {
            fprintf(stderr, "[Wiimote] Connection failed\n");
            wiiuse_cleanup(wiimotes, wiimoteCount);
            wiimotes = nullptr;
            wiimoteCount = 0;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        wiiuse_motion_sensing(wiimotes[0], 1);
        // Continuous reporting so accel/orientation update even without button presses.
        wiiuse_set_flags(wiimotes[0], WIIUSE_CONTINUOUS, 0);
        wiiuse_set_leds(wiimotes[0], WIIMOTE_LED_1);
        // Keep the poll loop responsive so disconnect() can interrupt promptly.
        wiiuse_set_timeout(wiimotes, wiimoteCount, 10, 10);
        // Report mode is managed internally by wiiuse based on enabled features.

        fprintf(stdout, "[Wiimote] Connected (auto-connect active)\n");
        connected = true;

        // Stay connected: poll until disconnect/unplug.
        while (!stopThread.load() && WIIMOTE_IS_CONNECTED(wiimotes[0])) {
            // wiiuse_poll blocks until an event or returns 0 quickly.
            int ev = wiiuse_poll(wiimotes, wiimoteCount);
            if (ev > 0) {
                readState();
            } else {
                // Keep state fresh even when no events arrive.
                readState();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
        }

        fprintf(stdout, "[Wiimote] Disconnected, will retry\n");
        connected = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            resetStateLocked();
        }
        wiiuse_cleanup(wiimotes, wiimoteCount);
        wiimotes = nullptr;
        wiimoteCount = 0;

        if (stopThread.load()) break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    autoConnecting = false;
}

void NunchukPublisher::update() {
    // The auto-connect loop already polls and refreshes the state. This method
    // exists to keep the calling convention and ensures the latest values are
    // available even when the loop is between polls.
    if (!connected.load()) return;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!wiimotes || wiimoteCount <= 0) return;
        // Refresh from the (possibly static) wiimote state.
        readStateLocked();
    }
}

void NunchukPublisher::readState() {
    std::lock_guard<std::mutex> lock(mutex);
    readStateLocked();
}

void NunchukPublisher::readStateLocked() {
    if (!wiimotes || wiimoteCount <= 0) return;
    wiimote_t* wm = wiimotes[0];
    if (!wm) return;

    bool isConn = WIIMOTE_IS_CONNECTED(wm) != 0;
    if (!isConn) {
        resetStateLocked();
        return;
    }

    state.connected = true;
    state.batteryLevel = wm->battery_level;

    state.led1 = WIIUSE_IS_LED_SET(wm, 1);
    state.led2 = WIIUSE_IS_LED_SET(wm, 2);
    state.led3 = WIIUSE_IS_LED_SET(wm, 3);
    state.led4 = WIIUSE_IS_LED_SET(wm, 4);

    state.buttons = wm->btns;

    // Wiimote accelerometer (raw calibrated bytes)
    state.accelX = wm->accel.x;
    state.accelY = wm->accel.y;
    state.accelZ = wm->accel.z;
    state.roll = wm->orient.roll;
    state.pitch = wm->orient.pitch;
    state.yaw = wm->orient.yaw;
    state.gforceX = wm->gforce.x;
    state.gforceY = wm->gforce.y;
    state.gforceZ = wm->gforce.z;

    // IR camera
    state.irEnabled = WIIUSE_USING_IR(wm) != 0;
    state.irNumDots = wm->ir.num_dots;
    state.irX = wm->ir.x;
    state.irY = wm->ir.y;
    state.irDistance = wm->ir.z;

    // Expansion (Nunchuk, with or without)
    state.expansionType = wm->exp.type;
    state.expansionConnected = WIIUSE_USING_EXP(wm) != 0;

    if (wm->exp.type == EXP_NUNCHUK || wm->exp.type == EXP_MOTION_PLUS_NUNCHUK) {
        const nunchuk_t& nc = wm->exp.nunchuk;
        state.joystickX = nc.js.x;
        state.joystickY = nc.js.y;
        state.joystickAngle = nc.js.ang;
        state.joystickMag = nc.js.mag;

        state.buttonC = (nc.btns & NUNCHUK_BUTTON_C) != 0;
        state.buttonZ = (nc.btns & NUNCHUK_BUTTON_Z) != 0;

        state.nunchukAccelX = nc.accel.x;
        state.nunchukAccelY = nc.accel.y;
        state.nunchukAccelZ = nc.accel.z;

        state.nunchukRoll = nc.orient.roll;
        state.nunchukPitch = nc.orient.pitch;
        state.nunchukYaw = nc.orient.yaw;
        state.nunchukGforceX = nc.gforce.x;
        state.nunchukGforceY = nc.gforce.y;
        state.nunchukGforceZ = nc.gforce.z;
    }
}

void NunchukPublisher::applyControls(EventManager* em, const Camera& cam, float deltaTime,
                                     ControllerManager* cm, Brush3dManager* brushManager) {
    if (!em || !cm) return;

    WiimoteState s = getState();
    if (!s.connected || !s.expansionConnected) return;
    if (s.expansionType != EXP_NUNCHUK && s.expansionType != EXP_MOTION_PLUS_NUNCHUK) return;

    ControllerContext& wctx = cm->wiimoteContext;
    const ControllerParameters& cp = *cm->getParameters();

    // ---- Initialize tracking state on first frame ----
    if (firstControlFrame) {
        startYaw = s.nunchukYaw;
        startPitch = s.nunchukPitch;
        startRoll = s.nunchukRoll;
        startAccelY = s.nunchukAccelY;
        firstControlFrame = false;
        prevButtons = s.buttons;
        prevC = s.buttonC;
        return;
    }

    // ---- Wiimote D-PAD page navigation (edge-triggered) ----
    uint16_t pressed = s.buttons & ~prevButtons;
    if (pressed & WIIMOTE_BUTTON_UP)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::WIIMOTE, PageNavigationEvent::Action::PREV_PAGE));
    if (pressed & WIIMOTE_BUTTON_DOWN)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::WIIMOTE, PageNavigationEvent::Action::NEXT_PAGE));
    if (pressed & WIIMOTE_BUTTON_LEFT)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::WIIMOTE, PageNavigationEvent::Action::PREV_SUBPAGE));
    if (pressed & WIIMOTE_BUTTON_RIGHT)
        em->publish(std::make_shared<PageNavigationEvent>(ControllerId::WIIMOTE, PageNavigationEvent::Action::NEXT_SUBPAGE));
    prevButtons = s.buttons;

    // Camera axes (all controls are camera-relative)
    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    float camAngDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float brushAngDeg = cp.cameraAngularSpeedDeg * deltaTime;

    ControllerAction action;

    // ---- Nunchuk Joystick: translation (both pages) ----
    {
        float jx = s.joystickX;
        float jy = s.joystickY;
        const float jdz = 0.15f;
        if (std::abs(jx) < jdz) jx = 0.0f;
        if (std::abs(jy) < jdz) jy = 0.0f;
        if (jx != 0.0f || jy != 0.0f) {
            float vel = 8.0f * deltaTime;
            if (jx != 0.0f) action.translate += right * (jx * vel * 32.0f);
            if (jy != 0.0f) action.translate += forward * (jy * vel * 32.0f);
        }
    }

    // ---- Nunchuk C button: capture start orientation on press ----
    // All accelerometer and YPR controls are gated by C.
    if (s.buttonC && !prevC) {
        startYaw = s.nunchukYaw;
        startPitch = s.nunchukPitch;
        startRoll = s.nunchukRoll;
        startAccelY = s.nunchukAccelY;
    }
    prevC = s.buttonC;

    auto wrap180 = [](float v) -> float {
        while (v > 180.0f) v -= 360.0f;
        while (v < -180.0f) v += 360.0f;
        return v;
    };

    const float inv90 = 1.0f / 90.0f;

    // ---- C-gated controls: translation, rotation, scale ----
    if (s.buttonC) {
        const PageCategory cat = wctx.activeCategory();
        const PageControl ctrl = wctx.activeControl();

        if (cat == PageCategory::CAMERA) {
            // YPR → camera rotation (axis-angle in camera-local frame)
            float yawOff = wrap180(s.nunchukYaw - startYaw);
            float pitchOff = wrap180(s.nunchukPitch - startPitch);
            float rollOff = wrap180(s.nunchukRoll - startRoll);
            const float rdz = 2.0f;
            if (std::abs(yawOff) > rdz)
                em->publish(std::make_shared<RotateCameraEvent>(cam.getUp(), yawOff * inv90 * camAngDeg));
            if (std::abs(pitchOff) > rdz)
                em->publish(std::make_shared<RotateCameraEvent>(cam.getRight(), -pitchOff * inv90 * camAngDeg));
            if (std::abs(rollOff) > rdz)
                em->publish(std::make_shared<RotateCameraEvent>(cam.getUp(), -rollOff * inv90 * camAngDeg));

        } else if (ctrl == PageControl::SCALE) {
            // Accelerometer Y offset from C-press capture → brush scale
            int scaleOff = s.nunchukAccelY - startAccelY;
            if (std::abs(scaleOff) > 2) {
                float scale = scaleOff * 0.001f;
                action.scaleDelta = glm::vec3(scale);
            }

        } else {
            // YPR → brush rotation
            float yawOff = wrap180(s.nunchukYaw - startYaw);
            float pitchOff = wrap180(s.nunchukPitch - startPitch);
            float rollOff = wrap180(s.nunchukRoll - startRoll);
            const float rdz = 2.0f;
            if (std::abs(yawOff) < rdz) yawOff = 0.0f;
            if (std::abs(pitchOff) < rdz) pitchOff = 0.0f;
            if (std::abs(rollOff) < rdz) rollOff = 0.0f;
            action.rotateDeg.x += yawOff * inv90 * brushAngDeg;
            action.rotateDeg.y += -pitchOff * inv90 * brushAngDeg;
            action.rotateDeg.z += rollOff * inv90 * brushAngDeg;
        }
    }

    // ---- Apply the action ----
    bool brushChanged = applyControllerAction(wctx, em, brushManager, action);
    if (brushChanged) em->queue(std::make_shared<RebuildBrushEvent>());
}

WiimoteState NunchukPublisher::getState() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state;
}
