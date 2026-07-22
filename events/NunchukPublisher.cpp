#include "NunchukPublisher.hpp"
#include "EventManager.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
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
#include <glm/gtc/constants.hpp>

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
    if (!s.connected) return;

    const ControllerParameters& cp = *cm->getParameters();

    // ---- Initialize tracking state on first frame ----
    if (firstControlFrame) {
        startYaw = s.nunchukYaw;
        startPitch = s.nunchukPitch;
        startRoll = s.nunchukRoll;
        startAccelY = s.nunchukAccelY;
        wiimoteStartYaw = s.yaw;
        wiimoteStartPitch = s.pitch;
        wiimoteStartRoll = s.roll;
        wiimoteStartAccelX = s.accelX;
        wiimoteStartAccelY = s.accelY;
        wiimoteStartAccelZ = s.accelZ;
        firstControlFrame = false;
        prevButtons = s.buttons;
        prevC = s.buttonC;
        prevB = (s.buttons & WIIMOTE_BUTTON_B) != 0;
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

    auto wrap180 = [](float v) -> float {
        while (v > 180.0f) v -= 360.0f;
        while (v < -180.0f) v += 360.0f;
        return v;
    };

    const float inv90 = 1.0f / 90.0f;

    // ========================================================================
    // NUNCHUK → CAMERA (when nunchuk is connected)
    // Joystick translates; C + YPR rotates.
    // ========================================================================
    if (s.expansionConnected &&
        (s.expansionType == EXP_NUNCHUK || s.expansionType == EXP_MOTION_PLUS_NUNCHUK)) {

        // ---- Nunchuk Joystick → camera translation ----
        {
            float jx = s.joystickX;
            float jy = s.joystickY;
            const float jdz = 0.15f;
            if (std::abs(jx) < jdz) jx = 0.0f;
            if (std::abs(jy) < jdz) jy = 0.0f;
            if (jx != 0.0f || jy != 0.0f) {
                glm::vec3 delta(0.0f);
                float vel = cam.speed * deltaTime;
                if (jx != 0.0f) delta += right * (jx * vel);
                if (jy != 0.0f) delta += forward * (jy * vel);
                em->publish(std::make_shared<TranslateCameraEvent>(delta));
            }
        }

        // ---- Nunchuk C button: capture start orientation on press ----
        if (s.buttonC && !prevC) {
            startYaw = s.nunchukYaw;
            startPitch = s.nunchukPitch;
            startRoll = s.nunchukRoll;
            startAccelY = s.nunchukAccelY;
        }
        prevC = s.buttonC;

        // ---- C held: YPR → camera rotation ----
        // Mapping: nunchuk roll → camera yaw (world Y), nunchuk yaw →
        // camera pitch (camera right), nunchuk pitch → camera roll
        // (camera forward) — this matches the natural grip of the nunchuk.
        if (s.buttonC) {
            float yawOff = wrap180(s.nunchukYaw - startYaw);
            float pitchOff = wrap180(s.nunchukPitch - startPitch);
            float rollOff = wrap180(s.nunchukRoll - startRoll);
            const float rdz = 2.0f;
            float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
            float y = (std::abs(rollOff) > rdz)  ? rollOff  * inv90 * angDeg : 0.0f;
            float p = (std::abs(yawOff) > rdz)   ? -yawOff  * inv90 * angDeg : 0.0f;
            float r = (std::abs(pitchOff) > rdz) ? -pitchOff * inv90 * angDeg : 0.0f;
            if (y != 0.0f || p != 0.0f || r != 0.0f)
                em->publish(std::make_shared<RotateCameraEvent>(y, p, r));
        }
    } else {
        // Nunchuk not connected — ensure C tracking resets cleanly
        prevC = false;
    }

    // ========================================================================
    // WIIMOTE → BRUSH (when B is pressed, works with or without nunchuk)
    // Accelerometer offsets → translation; YPR deltas → rotation.
    // ========================================================================
    bool bDown = (s.buttons & WIIMOTE_BUTTON_B) != 0;

    // Edge detection for B: capture initial orientation + accel
    if (bDown && !prevB) {
        wiimoteStartYaw = s.yaw;
        wiimoteStartPitch = s.pitch;
        wiimoteStartRoll = s.roll;
        wiimoteStartAccelX = s.accelX;
        wiimoteStartAccelY = s.accelY;
        wiimoteStartAccelZ = s.accelZ;
    }
    prevB = bDown;

    if (bDown && brushManager) {
        BrushEntry* be = brushManager->getSelectedEntry();
        if (be) {
            bool changed = false;

            // Accelerometer offsets → brush translation (camera-relative)
            {
                int offX = s.accelX - wiimoteStartAccelX;
                int offY = s.accelY - wiimoteStartAccelY;
                int offZ = s.accelZ - wiimoteStartAccelZ;
                const int adz = 3;
                if (std::abs(offX) > adz || std::abs(offY) > adz || std::abs(offZ) > adz) {
                    float vel = cp.wiimoteTransSpeed * deltaTime;
                    float scale = vel * 0.001f;
                    glm::vec3 trans(0.0f);
                    if (std::abs(offX) > adz) trans += right * (offX * scale);
                    if (std::abs(offY) > adz) trans += up     * (-offY * scale);
                    if (std::abs(offZ) > adz) trans += forward * (offZ * scale);
                    be->translate += trans;
                    changed = true;
                }
            }

            // YPR offsets → brush rotation
            {
                float yawOff = wrap180(s.yaw - wiimoteStartYaw);
                float pitchOff = wrap180(s.pitch - wiimoteStartPitch);
                float rollOff = wrap180(s.roll - wiimoteStartRoll);
                const float rdz = 2.0f;
                if (std::abs(yawOff) > rdz || std::abs(pitchOff) > rdz || std::abs(rollOff) > rdz) {
                    float rotSpeed = cp.wiimoteRotSpeed * deltaTime;
                    if (std::abs(yawOff) > rdz)   { be->yaw   += yawOff   * inv90 * rotSpeed; changed = true; }
                    if (std::abs(pitchOff) > rdz) { be->pitch -= pitchOff * inv90 * rotSpeed; changed = true; }
                    if (std::abs(rollOff) > rdz)  { be->roll  += rollOff  * inv90 * rotSpeed; changed = true; }
                }
            }

            if (changed) em->queue(std::make_shared<RebuildBrushEvent>());
        }
    }
}

WiimoteState NunchukPublisher::getState() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state;
}
