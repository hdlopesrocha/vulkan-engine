#pragma once

#include "NunchukState.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

struct wiimote_t;

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

    void autoConnectLoop();
    void readState();
    void readStateLocked();
    void resetStateLocked();
};
