#pragma once

// Serves a LocalScene's chunk ladder to remote clients over WebServer.
//
// Extracted from server.cpp so the same wiring serves both a headless scene
// (the standalone `bin/server`) and a live scene owned by the application
// (MyApp's settings toggle). The transport, protocol, snapshot and mesh
// builders stay in their own files (WebServer / ChunkProtocol / ChunkSnapshot /
// ChunkMesh); this class only binds them to one scene:
//
//   * REQUEST_ALL        -> chunksnap::collectAll (thread-safe, takes the tree lock)
//   * scene metadata     -> chunksnap::collectMeta (root lattice + ladder depth)
//   * REQUEST_MESH       -> chunkmesh::buildChunkMesh (one chunk's triangles)
//   * live octree edits  -> pushUpsert / pushDelete, fed by the caller from the
//                           scene's update handlers
//
// Nothing here depends on how the scene was produced: procedural generation,
// a scene file, or the running application's edited world all stream the same
// way. The class is safe to start/stop repeatedly and callable from any thread.

#include <atomic>
#include <cstdint>
#include <string>

#include "ChunkIds.hpp"
#include "WebServer.hpp"

class LocalScene;
struct OctreeNodeData;

class SceneServer {
public:
    struct Counters {
        uint64_t upserts = 0;
        uint64_t deletes = 0;
        uint64_t solidUpserts = 0;
        uint64_t transparentUpserts = 0;
        uint64_t solidDeletes = 0;
        uint64_t transparentDeletes = 0;
    };

    // `scene` must outlive the server (the providers reference it). The scene's
    // tree lock is taken inside the snapshot providers, so live edits are safe.
    SceneServer(LocalScene& scene, uint16_t port = 8080, bool verbose = true);
    ~SceneServer();

    SceneServer(const SceneServer&) = delete;
    SceneServer& operator=(const SceneServer&) = delete;

    // Starts the transport and wires the providers. Throws std::runtime_error
    // when the port cannot be bound (see lastError for the message).
    void start();
    void stop();
    bool running() const { return running_.load(); }

    uint16_t port() const { return web_.port(); }
    size_t clientCount() const { return web_.clientCount(); }
    const std::string& lastError() const { return error_; }

    // Live octree streaming. `layer` is the scene layer index the node belongs
    // to (0 = opaque terrain, 1 = transparent water by default). Safe from any
    // worker thread; no-ops when the server is not running.
    void pushUpsert(const OctreeNodeData& nd, int layer);
    void pushDelete(const OctreeNodeData& nd, int layer);

    Counters counters() const;

    // Chunk-id registry shared with the providers. Exposed for diagnostics and
    // for callers that want to collect a snapshot themselves (the standalone
    // server prints its chunk counts after loading).
    ChunkIdRegistry& ids() { return ids_; }

private:
    LocalScene& scene_;
    WebServer web_;
    ChunkIdRegistry ids_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> upserts_{0};
    std::atomic<uint64_t> deletes_{0};
    std::atomic<uint64_t> solidUpserts_{0};
    std::atomic<uint64_t> transparentUpserts_{0};
    std::atomic<uint64_t> solidDeletes_{0};
    std::atomic<uint64_t> transparentDeletes_{0};
    std::string error_;
    bool verbose_ = true;
};
