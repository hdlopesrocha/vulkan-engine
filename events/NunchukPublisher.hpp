#pragma once

#include "NunchukState.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

struct wiimote_t;
class EventManager;
class Camera;
class ControllerManager;
class Brush3dManager;
class Octree;

// Publishes the full state of a Wiimote (and any attached Nunchuk expansion)
// using the wiiuse library. Supports auto-connect with a retry loop so a
// Wiimote can be connected with or without a Nunchuk at any time.
class NunchukPublisher {
public:
    NunchukPublisher();
    ~NunchukPublisher();

    // Starts the background auto-connect loop. It will keep scanning and
    // reconnecting until disconnect() is called.
    void connect();
    void disconnect();

    // Polls the Wiimote and refreshes the published state. Call each frame.
    void update();

    // Maps Wiimote D-PAD and Nunchuk inputs to controller actions based on the
    // active page/subpage of the wiimote context. Must be called each frame
    // after update().
    void applyControls(EventManager* em, const Camera& cam, float deltaTime,
                       ControllerManager* cm, Brush3dManager* brushManager,
                       const Octree* octree = nullptr);

    WiimoteState getState() const;
    bool isAutoConnecting() const { return autoConnecting.load(); }
    bool isConnected() const { return state.connected; }

private:
    WiimoteState state;
    std::atomic<bool> autoConnecting = false;
    std::atomic<bool> stopThread = false;
    std::atomic<bool> connected = false;

    wiimote_t** wiimotes = nullptr;
    int wiimoteCount = 0;

    std::thread connectThread;
    mutable std::mutex mutex;

    // Tracking state for control mapping
    uint16_t prevButtons = 0;
    float startYaw = 0.0f;
    float startPitch = 0.0f;
    float startRoll = 0.0f;
    bool firstControlFrame = true;
    bool prevC = false;
    bool prevZ = false;
    bool prevA = false;

    // Whether wiiuse_set_motion_plus() has been called for the current connection.
    bool motionPlusEnabled = false;

    // Joystick acceleration timer (seconds since last centered)
    float joystickTimer = 0.0f;

    // Gyro bias tracking (leaky integrator, updated when A is not pressed)
    float gyroBiasYaw = 0.0f;
    float gyroBiasPitch = 0.0f;
    float gyroBiasRoll = 0.0f;

    // ── M+ Aim state ──
    bool aimWasActive = false;         // Aim subpage was active last frame
    float aimStartYaw = 0.0f;         // yaw when Aim page was entered (fallback)
    float aimStartPitch = 0.0f;       // pitch when Aim page was entered (fallback)
    float aimStartRoll = 0.0f;        // roll when Aim page was entered (fallback)
    glm::quat aimOrient = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // accumulated M+ orientation
    float aimGyroBiasYaw = 0.0f;      // gyro bias for aim integration
    float aimGyroBiasPitch = 0.0f;
    float aimGyroBiasRoll = 0.0f;

    void autoConnectLoop();
    void readState();
    void readStateLocked();
    void resetStateLocked();
};
