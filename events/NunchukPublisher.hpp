#pragma once

#include "NunchukState.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

struct wiimote_t;
class EventManager;
class Camera;
class ControllerManager;
class Brush3dManager;

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
                       ControllerManager* cm, Brush3dManager* brushManager);

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
    bool prevB = false;

    void autoConnectLoop();
    void readState();
    void readStateLocked();
    void resetStateLocked();
};
