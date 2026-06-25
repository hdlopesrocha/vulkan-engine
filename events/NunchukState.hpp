#pragma once

struct NunchukState {
    bool connected = false;

    // Joystick (normalized -1..1)
    float joystickX = 0.0f;
    float joystickY = 0.0f;

    // Buttons
    bool buttonC = false;
    bool buttonZ = false;

    // Raw accelerometer values (typical range ~80-200 at rest)
    int accelX = 0;
    int accelY = 0;
    int accelZ = 0;

    // Derived orientation (degrees)
    float roll = 0.0f;
    float pitch = 0.0f;
};
