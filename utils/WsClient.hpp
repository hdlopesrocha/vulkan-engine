#pragma once

// Minimal blocking WebSocket client (POSIX sockets, no third-party deps).
//
// Speaks the same binary+gzip framing as server/WebServer.hpp, from the
// other side of the socket: masked client frames, server handshake
// validation, fragmentation reassembly, ping auto-pong.
//
// Threading: one WsClient is NOT internally synchronized. RemoteScene gives
// it a single owner at a time (snapshot loop, then the receiver thread) and
// serializes sends with its own mutex.

#include <cstdint>
#include <string>
#include <vector>

class WsClient {
public:
    WsClient();
    ~WsClient();

    WsClient(const WsClient&) = delete;
    WsClient& operator=(const WsClient&) = delete;

    // TCP connect + WS handshake for path (default "/ws"). Returns false on
    // any failure (dns/tcp/timeout/bad status/accept-key mismatch); use
    // lastError() for the reason. timeoutMs covers the whole operation.
    bool connect(const std::string& host, uint16_t port,
                 const std::string& path = "/ws", int timeoutMs = 4000);
    void close();

    bool isOpen() const { return fd_ >= 0; }
    const std::string& lastError() const { return lastError_; }

    // Send one binary frame (payload should already be gzipped by the
    // caller). Returns false when the socket is dead.
    bool sendBinary(const uint8_t* data, size_t len);
    bool sendBinary(const std::vector<uint8_t>& v);

    // Read one full message (binary or text opcode; continuations reassembled,
    // pings answered). Returns false on close/timeout/error; closed() then
    // tells a clean server close apart from a transport failure.
    // timeoutMs < 0 blocks indefinitely.
    bool recvMessage(std::vector<uint8_t>& out, int timeoutMs = -1);
    bool closed() const { return sawClose_; }

private:
    int fd_ = -1;
    bool sawClose_ = false;
    std::string lastError_;

    bool readExact(uint8_t* out, size_t len, int timeoutMs);
    bool waitReadable(int timeoutMs);
};
