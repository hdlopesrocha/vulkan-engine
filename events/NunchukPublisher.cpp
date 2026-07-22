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

    ControllerContext& wctx = cm->wiimoteContext;
    const ControllerParameters& cp = *cm->getParameters();
    bool hasNunchuk = s.expansionConnected &&
        (s.expansionType == EXP_NUNCHUK || s.expansionType == EXP_MOTION_PLUS_NUNCHUK);

    // ---- Initialize tracking state on first frame ----
    if (firstControlFrame) {
        startYaw = s.yaw;
        startPitch = s.pitch;
        startRoll = s.roll;
        firstControlFrame = false;
        prevButtons = s.buttons;
        prevC = s.buttonC;
        prevZ = s.buttonZ;
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
    // NUNCHUK JOYSTICK → camera translation (forward/back + left/right)
    // ========================================================================
    if (hasNunchuk) {
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

    // ========================================================================
    // NUNCHUK C / Z → camera/brush vertical translation
    // C = translate UP (camera-relative), Z = translate DOWN
    // Both apply to camera (Camera page) or brush (Brush page).
    // ========================================================================
    if (hasNunchuk) {
        float vel = cam.speed * deltaTime;
        glm::vec3 upDelta = up * vel;
        glm::vec3 downDelta = up * (-vel);

        // C → UP (camera on Camera page, brush on Brush page)
        if (s.buttonC) {
            if (wctx.activeCategory() == PageCategory::CAMERA) {
                em->publish(std::make_shared<TranslateCameraEvent>(upDelta));
            } else if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    be->translate += upDelta;
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
        prevC = s.buttonC;

        // Z → DOWN (camera on Camera page, brush on Brush page)
        if (s.buttonZ) {
            if (wctx.activeCategory() == PageCategory::CAMERA) {
                em->publish(std::make_shared<TranslateCameraEvent>(downDelta));
            } else if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    be->translate += downDelta;
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
        prevZ = s.buttonZ;
    }

    // ========================================================================
    // WIIMOTE (MotionPlus) → ROTATION via B button
    // B + Wiimote YPR → camera/brush rotation based on active page.
    // ========================================================================
    bool bDown = (s.buttons & WIIMOTE_BUTTON_B) != 0;

    if (bDown && !prevB) {
        startYaw = s.yaw;
        startPitch = s.pitch;
        startRoll = s.roll;
    }
    prevB = bDown;

    if (bDown) {
        float yawOff = wrap180(s.yaw - startYaw);
        float pitchOff = wrap180(s.pitch - startPitch);
        float rollOff = wrap180(s.roll - startRoll);
        const float rdz = 2.0f;
        float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;

        if (wctx.activeCategory() == PageCategory::CAMERA) {
            // B + Wiimote YPR → camera rotation (yaw around world Y)
            float y = (std::abs(yawOff) > rdz)   ? yawOff   * inv90 * angDeg : 0.0f;
            float p = (std::abs(pitchOff) > rdz) ? -pitchOff * inv90 * angDeg : 0.0f;
            float r = (std::abs(rollOff) > rdz)  ? -rollOff  * inv90 * angDeg : 0.0f;
            if (y != 0.0f || p != 0.0f || r != 0.0f)
                em->publish(std::make_shared<RotateCameraEvent>(y, p, r));
        } else if (brushManager) {
            // B + Wiimote YPR → brush rotation
            BrushEntry* be = brushManager->getSelectedEntry();
            if (be) {
                float rotSpeed = cp.wiimoteRotSpeed * deltaTime;
                float dy = (std::abs(yawOff) > rdz)   ? yawOff   * inv90 * rotSpeed : 0.0f;
                float dp = (std::abs(pitchOff) > rdz) ? pitchOff * inv90 * rotSpeed : 0.0f;
                float dr = (std::abs(rollOff) > rdz)  ? rollOff  * inv90 * rotSpeed : 0.0f;
                if (dy != 0.0f || dp != 0.0f || dr != 0.0f) {
                    be->yaw   += dy;
                    be->pitch -= dp;
                    be->roll  += dr;
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
    }
}

WiimoteState NunchukPublisher::getState() const {
    std::lock_guard<std::mutex> lock(mutex);
    return state;
}
