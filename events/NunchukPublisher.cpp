#include "NunchukPublisher.hpp"

#ifdef HAS_CWIID
#include <bluetooth/bluetooth.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#endif

NunchukPublisher::NunchukPublisher() {}

NunchukPublisher::~NunchukPublisher() {
    disconnect();
}

void NunchukPublisher::connect() {
#ifdef HAS_CWIID
    if (connecting.load()) return;

    if (connectThread.joinable()) {
        connectThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (wiimote) {
            cwiid_close(wiimote);
            wiimote = nullptr;
        }
        state = NunchukState{};
    }

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

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (wiimote) {
            cwiid_close(wiimote);
            wiimote = nullptr;
        }
        state = NunchukState{};
    }
#else
    state = NunchukState{};
#endif
}

void NunchukPublisher::update() {
#ifdef HAS_CWIID
    std::lock_guard<std::mutex> lock(mutex);
    if (!wiimote) return;
    readState();
#endif
}

NunchukState NunchukPublisher::getState() const {
#ifdef HAS_CWIID
    std::lock_guard<std::mutex> lock(mutex);
#endif
    return state;
}

#ifdef HAS_CWIID
void NunchukPublisher::connectAsync() {
    bdaddr_t bdaddr = {{0, 0, 0, 0, 0, 0}};

    fprintf(stdout, "[Nunchuk] Scanning for Wiimote (press 1+2)...\n");

    cwiid_wiimote_t* wm = cwiid_open(&bdaddr, 0);
    if (!wm) {
        fprintf(stderr, "[Nunchuk] No Wiimote found or connection failed\n");
        connecting = false;
        return;
    }

    fprintf(stdout, "[Nunchuk] Wiimote connected\n");

    unsigned char rptMode = CWIID_RPT_BTN | CWIID_RPT_ACC | CWIID_RPT_NUNCHUK;
    if (cwiid_set_rpt_mode(wm, rptMode)) {
        fprintf(stderr, "[Nunchuk] Failed to set report mode\n");
        cwiid_close(wm);
        connecting = false;
        return;
    }

    if (cwiid_request_status(wm)) {
        fprintf(stderr, "[Nunchuk] Warning: cwiid_request_status failed\n");
    }

    // Poll briefly for the nunchuk extension to appear.
    // cwiid_get_state() returns the last received Wiimote data report,
    // which may not have arrived yet right after setting the report mode.
    bool foundNunchuk = false;
    for (int i = 0; i < 20; i++) {
        cwiid_state cs;
        if (cwiid_get_state(wm, &cs) == 0 && cs.ext_type == CWIID_EXT_NUNCHUK) {
            foundNunchuk = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (foundNunchuk) {
        fprintf(stdout, "[Nunchuk] Nunchuk extension detected\n");
    } else {
        fprintf(stdout, "[Nunchuk] Wiimote connected (no nunchuk detected yet, will keep polling)\n");
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (wiimote) {
            cwiid_close(wiimote);
        }
        wiimote = wm;
    }

    connecting = false;
}

void NunchukPublisher::readState() {
    cwiid_state cwState;
    if (cwiid_get_state(wiimote, &cwState)) return;

    if (cwState.ext_type == CWIID_EXT_NUNCHUK) {
        if (!state.connected) {
            fprintf(stdout, "[Nunchuk] Nunchuk extension detected\n");
        }
        state.connected = true;

        float rawX = static_cast<float>(cwState.ext.nunchuk.stick[0]);
        float rawY = static_cast<float>(cwState.ext.nunchuk.stick[1]);
        state.joystickX = (rawX - 128.0f) / 128.0f;
        state.joystickY = (rawY - 128.0f) / 128.0f;

        if (state.joystickX < -1.0f) state.joystickX = -1.0f;
        if (state.joystickX > 1.0f) state.joystickX = 1.0f;
        if (state.joystickY < -1.0f) state.joystickY = -1.0f;
        if (state.joystickY > 1.0f) state.joystickY = 1.0f;

        state.buttonC = (cwState.ext.nunchuk.buttons & CWIID_NUNCHUK_BTN_C) != 0;
        state.buttonZ = (cwState.ext.nunchuk.buttons & CWIID_NUNCHUK_BTN_Z) != 0;

        state.accelX = cwState.ext.nunchuk.acc[0];
        state.accelY = cwState.ext.nunchuk.acc[1];
        state.accelZ = cwState.ext.nunchuk.acc[2];

        float ax = static_cast<float>(state.accelX - 120);
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
    }
    // Don't set state.connected = false when ext_type doesn't match.
    // cwiid_get_state() returns the last received Wiimote data report,
    // which may not include extension info on the very first reads.
    // The nunchuk data will arrive in a subsequent frame.
}
#endif
