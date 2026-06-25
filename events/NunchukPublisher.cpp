#include "NunchukPublisher.hpp"

#ifdef HAS_CWIID
#include <bluetooth/bluetooth.h>
#include <cmath>
#include <cstdio>
#endif

NunchukPublisher::NunchukPublisher() {}

NunchukPublisher::~NunchukPublisher() {
    disconnect();
}

void NunchukPublisher::connect() {
#ifdef HAS_CWIID
    if (wiimote || connecting) return;
    connecting = true;
    connectThread = std::thread(&NunchukPublisher::connectAsync, this);
#else
    state = NunchukState{};
    state.connected = false;
#endif
}

void NunchukPublisher::disconnect() {
#ifdef HAS_CWIID
    connecting = false;
    if (connectThread.joinable()) {
        connectThread.join();
    }
    if (wiimote) {
        cwiid_close(wiimote);
        wiimote = nullptr;
    }
#endif
    state = NunchukState{};
}

void NunchukPublisher::update() {
#ifdef HAS_CWIID
    if (!wiimote) return;
    readState();
#endif
}

#ifdef HAS_CWIID
void NunchukPublisher::connectAsync() {
    bdaddr_t bdaddr = {{0, 0, 0, 0, 0, 0}};

    fprintf(stdout, "[Nunchuk] Scanning for Wiimote (press 1+2)...\n");

    wiimote = cwiid_open(&bdaddr, 0);
    if (!wiimote) {
        fprintf(stderr, "[Nunchuk] No Wiimote found or connection failed\n");
        connecting = false;
        return;
    }

    fprintf(stdout, "[Nunchuk] Wiimote connected\n");

    // Enable nunchuk extension data in reports
    unsigned char rptMode = CWIID_RPT_NUNCHUK;
    cwiid_set_rpt_mode(wiimote, rptMode);

    // Request status to get extension info
    cwiid_request_status(wiimote);

    connecting = false;
}

void NunchukPublisher::readState() {
    if (!wiimote) return;

    cwiid_state cwState;
    if (cwiid_get_state(wiimote, &cwState)) return;

    std::lock_guard<std::mutex> lock(stateMutex);

    if (cwState.ext.type == CWIID_EXT_NUNCHUK) {
        state.connected = true;

        // Joystick: raw 0..255, map to -1..1 (center ~128)
        float rawX = static_cast<float>(cwState.ext.nunchuk.stick[0]);
        float rawY = static_cast<float>(cwState.ext.nunchuk.stick[1]);
        state.joystickX = (rawX - 128.0f) / 128.0f;
        state.joystickY = (rawY - 128.0f) / 128.0f;

        // Clamp to [-1, 1]
        if (state.joystickX < -1.0f) state.joystickX = -1.0f;
        if (state.joystickX > 1.0f) state.joystickX = 1.0f;
        if (state.joystickY < -1.0f) state.joystickY = -1.0f;
        if (state.joystickY > 1.0f) state.joystickY = 1.0f;

        // Buttons
        state.buttonC = (cwState.ext.nunchuk.buttons & CWIID_NUNCHUK_BUTTON_C) != 0;
        state.buttonZ = (cwState.ext.nunchuk.buttons & CWIID_NUNCHUK_BUTTON_Z) != 0;

        // Raw accelerometer
        state.accelX = cwState.ext.nunchuk.acc[0];
        state.accelY = cwState.ext.nunchuk.acc[1];
        state.accelZ = cwState.ext.nunchuk.acc[2];

        // Derive roll and pitch from accelerometer
        // At rest: accelZ ~ 200 (gravity), accelX/accelY ~ 120
        // Normalize so that Z at rest in neutral orientation is ~1.0g
        float ax = static_cast<float>(state.accelX - 120); // bias to center
        float ay = static_cast<float>(state.accelY - 120);
        float az = static_cast<float>(state.accelZ - 120);

        float accelMag = std::sqrt(ax * ax + ay * ay + az * az);
        if (accelMag > 0.01f) {
            ax /= accelMag;
            ay /= accelMag;
            az /= accelMag;
            state.roll  = std::atan2(-ax, az) * 180.0f / static_cast<float>(M_PI);
            state.pitch = std::atan2(ay, std::sqrt(ax * ax + az * az)) * 180.0f / static_cast<float>(M_PI);
        }
    } else {
        state.connected = false;
    }
}
#endif
