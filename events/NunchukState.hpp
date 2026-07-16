#pragma once

#include <cstdint>

// Full state of a Wiimote and its attached expansion (e.g. Nunchuk).
// Exposed to the UI so all inputs and sensors can be visualized.
struct WiimoteState {
    bool connected = false;

    // ── Wiimote core ──
    bool led1 = false, led2 = false, led3 = false, led4 = false;
    float batteryLevel = 0.0f;   // 0..1
    uint16_t buttons = 0;        // WIIMOTE_BUTTON_* bitmask

    // Raw accelerometer (calibrated byte values ~0..255)
    int accelX = 0, accelY = 0, accelZ = 0;
    // Derived orientation (degrees, range -180..180)
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    // Gravity force vector (g)
    float gforceX = 0.0f, gforceY = 0.0f, gforceZ = 0.0f;

    // IR camera
    bool irEnabled = false;
    int irNumDots = 0;
    int irX = 0, irY = 0;       // absolute screen coordinates
    float irDistance = 0.0f;     // distance estimate

    // Expansion
    int expansionType = 0;       // EXP_NONE, EXP_NUNCHUK, ...
    bool expansionConnected = false;

    // ── Nunchuk expansion ──
    // Joystick (normalized -1..1)
    float joystickX = 0.0f;
    float joystickY = 0.0f;
    float joystickAngle = 0.0f; // degrees
    float joystickMag = 0.0f;    // 0..1

    // Buttons
    bool buttonC = false;
    bool buttonZ = false;

    // Raw accelerometer values (typical range ~0-255)
    int nunchukAccelX = 0;
    int nunchukAccelY = 0;
    int nunchukAccelZ = 0;

    // Derived orientation (degrees)
    float nunchukRoll = 0.0f;
    float nunchukPitch = 0.0f;
    float nunchukYaw = 0.0f;
    // Gravity force vector (g)
    float nunchukGforceX = 0.0f, nunchukGforceY = 0.0f, nunchukGforceZ = 0.0f;
};
