#pragma once

#include "NunchukState.hpp"

#ifdef HAS_CWIID
#include <cwiid.h>
#include <thread>
#include <mutex>
#include <atomic>
#endif

class NunchukPublisher {
public:
    NunchukPublisher();
    ~NunchukPublisher();

    void update();
    void connect();
    void disconnect();

    NunchukState getState() const;
    bool isConnecting() const { return connecting.load(); }

private:
    NunchukState state;
    std::atomic<bool> connecting = false;

#ifdef HAS_CWIID
    cwiid_wiimote_t* wiimote = nullptr;
    std::thread connectThread;
    mutable std::mutex mutex;

    void connectAsync();
    void readState();
#endif
};
