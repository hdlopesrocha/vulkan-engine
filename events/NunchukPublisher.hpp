#pragma once

#include "NunchukState.hpp"

#ifdef HAS_CWIID
#include <cwiid.h>
#include <thread>
#include <mutex>
#endif

class NunchukPublisher {
public:
    NunchukPublisher();
    ~NunchukPublisher();

    // Call each frame to poll nunchuk state
    void update();

    // Connect to a Wiimote with nunchuk (scans via Bluetooth)
    void connect();

    // Disconnect the Wiimote
    void disconnect();

    // Access current state
    const NunchukState& getState() const { return state; }

    // Connection status
    bool isConnecting() const { return connecting; }

private:
    NunchukState state;
    bool connecting = false;

#ifdef HAS_CWIID
    cwiid_wiimote_t* wiimote = nullptr;
    std::thread connectThread;
    std::mutex stateMutex;

    void connectAsync();
    void readState();
#endif
};
