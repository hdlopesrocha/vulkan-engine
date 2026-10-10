#pragma once

// Minimal HTTP + WebSocket server (POSIX sockets, no third-party deps).
//
// Routes:
//   GET / or /index.html  -> serves server/web/index.html (the Vue viewer)
//   GET /health           -> {"status":"ok","chunks":N,"clients":N}
//   GET /ws               -> WebSocket upgrade; binary gzipped chunk protocol
//
// Every WebSocket binary frame payload is a gzip blob (see ChunkProtocol).
// Client sends gzipped [0x01] to request a full snapshot (or gzipped
// [0x02 + u64 id] to request one chunk's tessellated mesh); the server
// streams live upserts/deletes as the octree changes.

#include "ChunkProtocol.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class WebServer {
public:
    explicit WebServer(uint16_t port = 8080);
    ~WebServer();

    WebServer(const WebServer&) = delete;
    WebServer& operator=(const WebServer&) = delete;

    // Non-blocking: spawns accept + flusher threads.
    void start();
    void stop();

    uint16_t port() const { return port_; }

    // Octree callbacks (thread-safe, callable from any worker thread).
    void pushUpsert(const chunkproto::ChunkRecord& rec);
    void pushDelete(uint64_t id);

    // Snapshot provider for REQUEST_ALL (must be thread-safe).
    void setSnapshotProvider(std::function<std::vector<chunkproto::ChunkRecord>()> fn);

    // Scene-meta provider for REQUEST_ALL (must be thread-safe). Sent once
    // per snapshot right after SNAPSHOT_START.
    void setMetaProvider(std::function<std::vector<chunkproto::SceneMetaLayer>()> fn);

    // Mesh provider for REQUEST_MESH (must be thread-safe). Returns true when
    // the id names a live chunk (mesh may be empty for non-surface chunks).
    void setMeshProvider(std::function<bool(uint64_t, chunkproto::MeshData&)> fn);

    size_t clientCount() const;
    uint64_t upsertsSent() const;
    uint64_t deletesSent() const;
    uint64_t meshesSent() const;

private:
    uint16_t port_;
    int listenFd_ = -1;
    bool running_ = false;

    void acceptLoop();
    void flusherLoop();
    void clientThread(int fd);
    bool handleHttpOrWs(int fd);
    void wsLoop(int fd);
    void sendSnapshotTo(int fd);
    void sendMeshTo(int fd, uint64_t chunkId);

    void sendRaw(int fd, const std::vector<uint8_t>& gzipped, uint8_t opcode);
    void sendGzipped(int fd, const std::vector<uint8_t>& raw, uint8_t opcode);
    void broadcastRaw(const std::vector<uint8_t>& gzipped, uint8_t opcode);

    // --- state ---
    mutable std::mutex clientsMutex_;
    std::vector<int> clients_; // active WS fds
    std::mutex sendMutex_;     // serializes whole-frame sends

    std::mutex pendingMutex_;
    std::vector<chunkproto::ChunkRecord> pendingUpserts_;
    std::vector<uint64_t> pendingDeletes_;

    // Serializes whole streams (snapshot START..END vs one flusher flush) so a
    // live broadcast can never interleave inside another client's snapshot.
    // Individual frame writes are still protected by sendMutex_ (lock order:
    // streamMutex_ -> sendMutex_).
    std::mutex streamMutex_;

    std::function<std::vector<chunkproto::ChunkRecord>()> snapshotProvider_;
    std::function<std::vector<chunkproto::SceneMetaLayer>()> metaProvider_;
    std::function<bool(uint64_t, chunkproto::MeshData&)> meshProvider_;

    std::thread acceptThread_;
    std::thread flusherThread_;

    std::atomic<uint64_t> upsertsSent_{0};
    std::atomic<uint64_t> deletesSent_{0};
    std::atomic<uint64_t> meshesSent_{0};

    std::string indexHtml_; // cached page
    std::mutex indexMutex_;
    std::string loadIndexHtml();
};
