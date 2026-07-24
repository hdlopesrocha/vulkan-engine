#include "NunchukPublisher.hpp"
#include "EventManager.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "../math/Camera.hpp"
#include "../math/Ray.hpp"
#include "../space/Octree.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include "RebuildBrushEvent.hpp"

#include <wiiuse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
        motionPlusEnabled = false;

        // Stay connected: poll until disconnect/unplug.
        while (!stopThread.load() && WIIMOTE_IS_CONNECTED(wiimotes[0])) {
            // wiiuse_poll blocks until an event or returns 0 quickly.
            int ev = wiiuse_poll(wiimotes, wiimoteCount);

            // Activate MotionPlus once the expansion handshake is complete.
            wiimote_t* wm = wiimotes[0];
            if (!motionPlusEnabled && wm
                && WIIMOTE_IS_SET(wm, WIIMOTE_STATE_MPLUS_PRESENT)
                && !WIIMOTE_IS_SET(wm, WIIMOTE_STATE_EXP_HANDSHAKE))
            {
                int mpStatus = (wm->exp.type != EXP_NONE) ? 2 : 1;
                wiiuse_set_motion_plus(wm, mpStatus);
                motionPlusEnabled = true;
                fprintf(stdout, "[Wiimote] MotionPlus activated (status=%d)\n", mpStatus);
            }

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

    // MotionPlus gyroscope rates
    state.hasMotionPlus = (wm->exp.type == EXP_MOTION_PLUS || wm->exp.type == EXP_MOTION_PLUS_NUNCHUK);
    if (state.hasMotionPlus) {
        state.gyroYawRate   = wm->exp.mp.angle_rate_gyro.yaw;
        state.gyroPitchRate = wm->exp.mp.angle_rate_gyro.pitch;
        state.gyroRollRate  = wm->exp.mp.angle_rate_gyro.roll;
    }

    // Don't read nunchuk state while M+ handshake is in progress
    // (wm->exp.type is still EXP_NUNCHUK but Wiimote already sends M+-interleaved data)
    bool mplusHShake = motionPlusEnabled &&
        WIIMOTE_IS_SET(wm, WIIMOTE_STATE_EXP_HANDSHAKE);

    if ((wm->exp.type == EXP_NUNCHUK || wm->exp.type == EXP_MOTION_PLUS_NUNCHUK) && !mplusHShake) {
        const nunchuk_t& nc = wm->exp.nunchuk;

        // motion_plus_event() calls calc_joystick_state() → normalized [-1,1].
        // During M+ activation transition the nunchuk handle may have written
        // raw ADC while processing M+ frames → detect and re-normalize.
        float jx = nc.js.x;
        float jy = nc.js.y;
        float ang = nc.js.ang;
        float mag = nc.js.mag;
        if (jx > 1.0f || jx < -1.0f || jy > 1.0f || jy < -1.0f) {
            float cx = (nc.js.center.x > 0) ? nc.js.center.x : 128.0f;
            float cy = (nc.js.center.y > 0) ? nc.js.center.y : 128.0f;
            float rx = (nc.js.max.x > nc.js.min.x) ? (nc.js.max.x - nc.js.min.x) / 2.0f : 128.0f;
            float ry = (nc.js.max.y > nc.js.min.y) ? (nc.js.max.y - nc.js.min.y) / 2.0f : 128.0f;
            jx = (jx - cx) / rx;
            jy = (jy - cy) / ry;
            if (jx < -1.0f) jx = -1.0f; else if (jx > 1.0f) jx = 1.0f;
            if (jy < -1.0f) jy = -1.0f; else if (jy > 1.0f) jy = 1.0f;
            ang = std::atan2(jy, jx);
            mag = std::sqrt(jx * jx + jy * jy);
        }
        state.joystickX = jx;
        state.joystickY = jy;
        state.joystickAngle = ang;
        state.joystickMag = mag;

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
    } else {
        state.joystickX = 0.0f;
        state.joystickY = 0.0f;
        state.joystickAngle = 0.0f;
        state.joystickMag = 0.0f;
        state.buttonC = false;
        state.buttonZ = false;
        state.nunchukAccelX = 0.0f;
        state.nunchukAccelY = 0.0f;
        state.nunchukAccelZ = 0.0f;
        state.nunchukRoll = 0.0f;
        state.nunchukPitch = 0.0f;
        state.nunchukYaw = 0.0f;
        state.nunchukGforceX = 0.0f;
        state.nunchukGforceY = 0.0f;
        state.nunchukGforceZ = 0.0f;
    }
}

void NunchukPublisher::applyControls(EventManager* em, const Camera& cam, float deltaTime,
                                     ControllerManager* cm, Brush3dManager* brushManager,
                                     const Octree* octree) {
    if (!em || !cm) return;

    WiimoteState s = getState();
    if (!s.connected) return;

    // DEBUG: print state every 2s
    {
        static auto lastPrint = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now - lastPrint > std::chrono::seconds(2)) {
            lastPrint = now;
            fprintf(stderr, "[Wiimote] type=%d jx=%.3f jy=%.3f btns=0x%04x exp=%d "
                    "C=%d Z=%d gyroY=%.1f gyroP=%.1f gyroR=%.1f\n",
                    s.expansionType, s.joystickX, s.joystickY, s.buttons,
                    s.expansionConnected,
                    s.buttonC, s.buttonZ,
                    s.gyroYawRate, s.gyroPitchRate, s.gyroRollRate);
        }
    }

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
        prevA = (s.buttons & WIIMOTE_BUTTON_A) != 0;
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

    // Wiimote +/- → cycle brush SDF primitive type (only on Brush page)
    if (brushManager) {
        if (pressed & WIIMOTE_BUTTON_PLUS) {
            if (wctx.activeCategory() == PageCategory::BRUSH) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    be->sdfType = (be->sdfType + 1) % 10;
                    fprintf(stderr, "[Wiimote] Brush SDF type -> %d\n", be->sdfType);
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
        if (pressed & WIIMOTE_BUTTON_MINUS) {
            if (wctx.activeCategory() == PageCategory::BRUSH) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    be->sdfType = (be->sdfType - 1 + 10) % 10;
                    fprintf(stderr, "[Wiimote] Brush SDF type -> %d\n", be->sdfType);
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
    }

    // Wiimote 1 → snap reset / brush place
    if (pressed & WIIMOTE_BUTTON_ONE) {
        if (brushManager) {
            BrushEntry* be = brushManager->getSelectedEntry();
            if (be) {
                const ControllerPage* sub = wctx.activeSubpage();
                if (sub && sub->control == PageControl::AIM) {
                    // Aim subpage: reset snap offset
                    be->snapTranslation = glm::vec3(0.0f);
                    fprintf(stderr, "[Wiimote] snapTranslation reset\n");
                } else if (wctx.activeCategory() == PageCategory::BRUSH) {
                    // Brush page: place brush at camera + forward * 4 * scaleLen
                    float scaleLen = glm::length(be->scale);
                    be->translate = cam.getPosition() + cam.getForward() * (4.0f * scaleLen);
                    fprintf(stderr, "[Wiimote] brush placed at %.1f %.1f %.1f\n",
                            be->translate.x, be->translate.y, be->translate.z);
                }
                em->queue(std::make_shared<RebuildBrushEvent>());
            }
        }
    }

    prevButtons = s.buttons;

    // Camera axes (all controls are camera-relative)
    glm::vec3 right = cam.getRight();
    glm::vec3 up = cam.getUp();
    glm::vec3 forward = cam.getForward();

    // Joystick acceleration: exponential ramp from 0 (at t=0s) → 1 (at t=1s) → grows after
    float joystickAccel = 0.0f;
    if (hasNunchuk) {
        float ax = s.joystickX;
        float ay = s.joystickY;
        if (!std::isfinite(ax)) ax = 0.0f;
        if (!std::isfinite(ay)) ay = 0.0f;
        ax = glm::clamp(ax, -1.0f, 1.0f);
        ay = glm::clamp(ay, -1.0f, 1.0f);
        const float ajdz = 0.30f;
        if (std::abs(ax) < ajdz) ax = 0.0f;
        if (std::abs(ay) < ajdz) ay = 0.0f;
        if (ax != 0.0f || ay != 0.0f) {
            joystickTimer += deltaTime;
            joystickAccel = (std::exp(joystickTimer) - 1.0f) / (std::exp(1.0f) - 1.0f);
        } else {
            joystickTimer = 0.0f;
        }
    }

    auto wrap180 = [](float v) -> float {
        while (v > 180.0f) v -= 360.0f;
        while (v < -180.0f) v += 360.0f;
        return v;
    };

    // ========================================================================
    // NUNCHUK JOYSTICK → camera translation (forward/back + left/right)
    // ========================================================================
    if (hasNunchuk) {
        float jx = s.joystickX;
        float jy = s.joystickY;
        // Guard against garbage from uninitialized / transition states
        if (!std::isfinite(jx)) jx = 0.0f;
        if (!std::isfinite(jy)) jy = 0.0f;
        jx = glm::clamp(jx, -1.0f, 1.0f);
        jy = glm::clamp(jy, -1.0f, 1.0f);
        const float jdz = 0.30f;
        if (std::abs(jx) < jdz) jx = 0.0f;
        if (std::abs(jy) < jdz) jy = 0.0f;
        if (jx != 0.0f || jy != 0.0f) {
            float vel = cam.speed * deltaTime * joystickAccel;
            if (wctx.activeCategory() == PageCategory::CAMERA) {
                glm::vec3 delta(0.0f);
                if (jx != 0.0f) delta += right * (jx * vel);
                if (jy != 0.0f) delta += forward * (jy * vel);
                em->publish(std::make_shared<TranslateCameraEvent>(delta));
            } else if (brushManager) {
                // Check that we are NOT on the Aim subpage (Aim handles nunchuk
                // as a relative shift from the intersection point instead).
                const ControllerPage* bp = wctx.activeSubpage();
                if (!bp || bp->control != PageControl::AIM) {
                    BrushEntry* be = brushManager->getSelectedEntry();
                    if (be) {
                        if (jx != 0.0f) be->translate += right * (jx * vel);
                        if (jy != 0.0f) be->translate += forward * (jy * vel);
                        em->queue(std::make_shared<RebuildBrushEvent>());
                    }
                }
            }
        }
    }

    // ========================================================================
    // NUNCHUK C / Z → camera/brush vertical translation
    // C = translate UP (camera-relative), Z = translate DOWN
    // Both apply to camera (Camera page) or brush (Brush page).
    // ========================================================================
    if (hasNunchuk) {
        // ── Check if Aim subpage is active (C/Z are handled as shift below) ──
        const ControllerPage* aimCheck = wctx.activeSubpage();
        bool aimActive = aimCheck && aimCheck->control == PageControl::AIM;

        float vel = cam.speed * deltaTime;
        glm::vec3 upDelta = up * vel;
        glm::vec3 downDelta = up * (-vel);

        // C → UP (camera on Camera page, brush on Brush page, skip on Aim)
        if (s.buttonC) {
            if (wctx.activeCategory() == PageCategory::CAMERA) {
                em->publish(std::make_shared<TranslateCameraEvent>(upDelta));
            } else if (brushManager && !aimActive) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    be->translate += upDelta;
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }
        prevC = s.buttonC;

        // Z → DOWN (camera on Camera page, brush on Brush page, skip on Aim)
        if (s.buttonZ) {
            if (wctx.activeCategory() == PageCategory::CAMERA) {
                em->publish(std::make_shared<TranslateCameraEvent>(downDelta));
            } else if (brushManager && !aimActive) {
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
    // WIIMOTE (MotionPlus) → ROTATION via A button
    // A + Wiimote YPR → camera/brush rotation based on active page.
    // ========================================================================
    bool aDown = (s.buttons & WIIMOTE_BUTTON_A) != 0;

    if (aDown && !prevA) {
        startYaw = s.yaw;
        startPitch = s.pitch;
        startRoll = s.roll;
    }
    prevA = aDown;

    if (aDown) {
        const float rdz = 2.0f;

        // Determine rotation input: gyro rates (MotionPlus) or YPR offsets
        float yawInput, pitchInput, rollInput;

        if (s.hasMotionPlus) {
            // Gyro bias: update leaky integrator when A NOT pressed,
            // subtract bias when A IS pressed (eliminates resting drift)
            if (!aDown) {
                gyroBiasYaw   += (s.gyroYawRate   - gyroBiasYaw)   * 0.02f;
                gyroBiasPitch += (s.gyroPitchRate - gyroBiasPitch) * 0.02f;
                gyroBiasRoll  += (s.gyroRollRate  - gyroBiasRoll)  * 0.02f;
            }
            float gy = std::isfinite(s.gyroYawRate)   ? glm::clamp(s.gyroYawRate   - gyroBiasYaw,   -720.0f, 720.0f)   : 0.0f;
            float gp = std::isfinite(s.gyroPitchRate) ? glm::clamp(s.gyroPitchRate - gyroBiasPitch, -720.0f, 720.0f) : 0.0f;
            float gr = std::isfinite(s.gyroRollRate)  ? glm::clamp(s.gyroRollRate  - gyroBiasRoll,  -720.0f, 720.0f)  : 0.0f;
            yawInput   = (std::abs(gy) > rdz) ? gy : 0.0f;
            pitchInput = (std::abs(gp) > rdz) ? gp : 0.0f;
            rollInput  = (std::abs(gr) > rdz) ? gr : 0.0f;
        } else {
            float yawOff   = wrap180(s.yaw - startYaw);
            float pitchOff = wrap180(s.pitch - startPitch);
            float rollOff  = wrap180(s.roll - startRoll);
            yawInput   = (std::abs(yawOff) > rdz)   ? yawOff   : 0.0f;
            pitchInput = (std::abs(pitchOff) > rdz) ? pitchOff : 0.0f;
            rollInput  = (std::abs(rollOff) > rdz)  ? rollOff  : 0.0f;
        }

        if (wctx.activeCategory() == PageCategory::CAMERA) {
            // Camera uses its own angular speed setting for sensitivity
            float camScale = glm::degrees(cam.angularSpeedRad) * deltaTime / 90.0f;
            float y =  yawInput * camScale;
            float p = -pitchInput * camScale;
            float r = -rollInput * camScale;
            if (y != 0.0f || p != 0.0f || r != 0.0f)
                em->publish(std::make_shared<RotateCameraEvent>(y, p, r));
        } else if (brushManager) {
            // Do NOT apply rotation when on the Aim subpage (it has its own
            // gyro integration).
            const ControllerPage* rotSub = wctx.activeSubpage();
            if (!rotSub || rotSub->control != PageControl::AIM) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) {
                    float brushScale = deltaTime * cp.wiimoteRotSpeed / 90.0f;
                    float dy = yawInput   * brushScale;
                    float dp = pitchInput * brushScale;
                    float dr = rollInput  * brushScale;
                    if (dy != 0.0f || dp != 0.0f || dr != 0.0f) {
                        glm::quat q = be->rot;
                        glm::quat camOrient = cam.getOrientation();
                        glm::quat rCam = glm::angleAxis(glm::radians(dy),  glm::vec3(0.0f, 1.0f, 0.0f))
                                       * glm::angleAxis(glm::radians(-dp), glm::vec3(1.0f, 0.0f, 0.0f))
                                       * glm::angleAxis(glm::radians(-dr), glm::vec3(0.0f, 0.0f, -1.0f));
                        glm::quat rWorld = camOrient * rCam * glm::conjugate(camOrient);
                        be->rot = glm::normalize(rWorld * q);
                        em->queue(std::make_shared<RebuildBrushEvent>());
                    }
                }
            }
        }
    }

    // ========================================================================
    // AIM SUBPAGE: M+ orientation → ray → leaf intersection → brush position
    // The nunchuk joystick always accumulates into snapTranslation (independent
    // of the A button); C/Z shift it vertically.  Brush translate is set to
    // hitPos + snapTranslation only when the ray hits — no hit = no change.
    //
    // Gyro integration follows the same pattern as the rotation section above:
    //   • A button must be held to accumulate M+ rotation into aimOrient
    //   • Bias is updated via leaky integrator when NOT actively aiming
    //   • Fallback to absolute YPR offsets when M+ is unavailable
    // ========================================================================
    {
        const ControllerPage* subpage = wctx.activeSubpage();
        bool onAim = subpage && subpage->control == PageControl::AIM;

        // ── Gyro bias: update when not actively aiming ──
        if (!onAim || !aDown) {
            aimGyroBiasYaw   += (s.gyroYawRate   - aimGyroBiasYaw)   * 0.02f;
            aimGyroBiasPitch += (s.gyroPitchRate - aimGyroBiasPitch) * 0.02f;
            aimGyroBiasRoll  += (s.gyroRollRate  - aimGyroBiasRoll)  * 0.02f;
        }

        if (onAim && !aimWasActive) {
            // Entering Aim page: reset orientation to identity (forward)
            aimOrient = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            aimStartYaw = s.yaw;
            aimStartPitch = s.pitch;
            aimStartRoll = s.roll;
        }
        aimWasActive = onAim;

        // ── Nunchuk adjusts snapTranslation in Aim mode ──
        if (onAim && brushManager) {
            BrushEntry* be = brushManager->getSelectedEntry();
            if (be) {
                if (hasNunchuk) {
                    float jx = s.joystickX;
                    float jy = s.joystickY;
                    if (!std::isfinite(jx)) jx = 0.0f;
                    if (!std::isfinite(jy)) jy = 0.0f;
                    jx = glm::clamp(jx, -1.0f, 1.0f);
                    jy = glm::clamp(jy, -1.0f, 1.0f);
                    const float jdz = 0.30f;
                    if (std::abs(jx) < jdz) jx = 0.0f;
                    if (std::abs(jy) < jdz) jy = 0.0f;
                    if (jx != 0.0f || jy != 0.0f) {
                        float vel = cam.speed * deltaTime * joystickAccel;
                        glm::vec3 delta(0.0f);
                        if (jx != 0.0f) delta += right * (jx * vel);
                        if (jy != 0.0f) delta += forward * (jy * vel);
                        be->snapTranslation += delta;
                        be->translate += delta;
                        em->queue(std::make_shared<RebuildBrushEvent>());
                    }
                }
                if (s.buttonC) {
                    be->snapTranslation += up * (cam.speed * deltaTime);
                    be->translate += up * (cam.speed * deltaTime);
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
                if (s.buttonZ) {
                    be->snapTranslation -= up * (cam.speed * deltaTime);
                    be->translate -= up * (cam.speed * deltaTime);
                    em->queue(std::make_shared<RebuildBrushEvent>());
                }
            }
        }

        // ── M+ aim-direction integration (A-only) ──
        if (onAim && aDown) {
            const float rdz = 2.0f;

            if (s.hasMotionPlus) {
                float gy = std::isfinite(s.gyroYawRate)   ? glm::clamp(s.gyroYawRate   - aimGyroBiasYaw,   -720.0f, 720.0f)   : 0.0f;
                float gp = std::isfinite(s.gyroPitchRate) ? glm::clamp(s.gyroPitchRate - aimGyroBiasPitch, -720.0f, 720.0f) : 0.0f;
                float gr = std::isfinite(s.gyroRollRate)  ? glm::clamp(s.gyroRollRate  - aimGyroBiasRoll,  -720.0f, 720.0f)  : 0.0f;
                float dy = (std::abs(gy) > rdz) ? gy * deltaTime : 0.0f;
                float dp = (std::abs(gp) > rdz) ? gp * deltaTime : 0.0f;
                float dr = (std::abs(gr) > rdz) ? gr * deltaTime : 0.0f;

                if (dy != 0.0f || dp != 0.0f || dr != 0.0f) {
                    glm::quat dq = glm::angleAxis(glm::radians(dy),  glm::vec3(0.0f, 1.0f, 0.0f))
                                 * glm::angleAxis(glm::radians(-dp), glm::vec3(1.0f, 0.0f, 0.0f))
                                 * glm::angleAxis(glm::radians(-dr), glm::vec3(0.0f, 0.0f, -1.0f));
                    aimOrient = glm::normalize(dq * aimOrient);
                }
            } else {
                float dy = wrap180(s.yaw   - aimStartYaw);
                float dp = wrap180(s.pitch - aimStartPitch);
                float dr = wrap180(s.roll  - aimStartRoll);
                dy = (std::abs(dy) > rdz) ? glm::radians(dy) : 0.0f;
                dp = (std::abs(dp) > rdz) ? glm::radians(-dp) : 0.0f;
                dr = (std::abs(dr) > rdz) ? glm::radians(-dr) : 0.0f;

                if (dy != 0.0f || dp != 0.0f || dr != 0.0f) {
                    glm::quat qAim = glm::angleAxis(dy, glm::vec3(0.0f, 1.0f, 0.0f))
                                   * glm::angleAxis(dp, glm::vec3(1.0f, 0.0f, 0.0f))
                                   * glm::angleAxis(dr, glm::vec3(0.0f, 0.0f, -1.0f));
                    aimOrient = qAim;
                }
            }
        }

        // ── Ray-cast (always in Aim mode, direction from aimOrient) ──
        if (onAim && octree && brushManager) {
            BrushEntry* be = brushManager->getSelectedEntry();
            if (be) {
                glm::vec3 V_cam = glm::normalize(aimOrient * glm::vec3(0.0f, 0.0f, -1.0f));
                glm::vec3 V_world = glm::normalize(cam.getOrientation() * V_cam);
                Ray ray(cam.getPosition(), V_world);

                glm::vec3 hitPos;
                if (octree->intersect(ray, hitPos)) {
                    be->translate = hitPos + be->snapTranslation;
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
