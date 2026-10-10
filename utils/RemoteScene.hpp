#pragma once

// RemoteScene: a Scene implementation backed by the chunk-streaming server
// instead of local octrees.
//
// MyApp tries this first at startup; when the WebSocket is unreachable it
// falls back to LocalScene (no local state is built in remote mode, so the
// fallback is clean).
//
// Protocol (see server/ChunkProtocol.hpp): the client only ever learns
// opaque chunk ids, bounding boxes + attributes, and tessellated geometry.
// Chunk ids are session-scoped monotonic numbers minted by the server —
// never node pointers (see server/ChunkIds.hpp). The client's own camera
// never leaves the process.
//
// How it feeds the renderer:
//   - loadScene() ignores the SceneLoaderCallback (there is no local octree
//     to build), downloads the snapshot (START/META/BATCH.../END) and replays
//     every chunk record into the given update handlers with a locally-owned
//     stub OctreeNode, then starts the receiver thread for live batches.
//   - The renderer works on stub nodes exactly like octree nodes: NodeID is
//     the stub address, which is stable for the connection lifetime (stubs
//     are never freed until disconnect, so deferred-slot bookkeeping in
//     MyApp/SceneRenderer cannot alias a recycled address).
//   - requestModel3D() fetches MESH_DATA for the stub's server id over the
//     same connection (serialized, matched by reply id) and rebuilds a
//     Geometry with computed smooth normals + triplanar UVs. Results are
//     cached by (serverId, version), so re-dispatches are free.
//   - requestSDFCubes()/requestBoundingBoxes() synthesize one debug cube per
//     chunk from the record (coarser than the local per-node walk, but the
//     debug overlays consume them unchanged).
//
// Threading: the receiver thread only records into the (thread-safe)
// UniqueChangeCollector lambdas received by loadScene; MyApp dispatches on
// its tessellation threads exactly like the local path. Mesh fetches run on
// the caller's worker thread with a 30 s timeout; on failure the event is
// re-recorded (throttled per chunk) so a later dispatch retries.

#include "Scene.hpp"
#include "WsClient.hpp"
#include "../math/BoundingCube.hpp"
#include "../space/OctreeNode.hpp"
// Wire codec (server/ is compiled into the app too; the protocol is shared).
#include "../server/ChunkProtocol.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class RemoteScene : public Scene {
public:
    struct Endpoint {
        std::string host = "127.0.0.1";
        uint16_t port = 8080;
    };

    RemoteScene();
    ~RemoteScene();

    RemoteScene(const RemoteScene&) = delete;
    RemoteScene& operator=(const RemoteScene&) = delete;

    // TCP + WS handshake only (fast fail). False = keep LocalScene.
    bool connect(const Endpoint& ep, int timeoutMs = 4000);
    bool isConnected() const { return connected_.load(); }
    // Idempotent. Wakes fetchers, joins the receiver thread.
    void disconnect();
    bool failed() const { return failed_.load(); } // set when socket dies mid-loadScene
    // Scene interface --------------------------------------------------
    // Dynamic layers: handlers are indexed by Layer (vector position).
    void loadScene(SceneLoaderCallback& callback,
                   std::vector<Octree::OctreeNodeDataHandler> updateHandlers,
                   std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) override;
    void requestModel3D(Layer layer, OctreeNodeData& data,
                        const GeometryLodCallback& callback,
                        ThreadPool* poolOverride = nullptr) override;
    void requestSDFCubes(Layer layer, OctreeNodeData& data,
                         const SdfCubeCallback& callback,
                         ThreadPool* poolOverride = nullptr) override;
    void requestBoundingBoxes(Layer layer, OctreeNodeData& data,
                              const BBoxCallback& callback,
                              ThreadPool* poolOverride = nullptr) override;
    bool isNodeUpToDate(Layer layer, OctreeNodeData& data, uint version) override;
    int maxChunkLod(Layer layer, float minSize) const override;
    glm::vec3 lodRootMin(Layer layer) const override;

    // ── Dynamic layer list (mirrors the server's layer ids) ──
    size_t layerCount() const override;
    std::string layerName(Layer layer) const override;
    void setLayerName(Layer layer, const std::string& name) override;
    LayerRendererType layerRenderer(Layer layer) const override;
    void setLayerRenderer(Layer layer, LayerRendererType renderer) override;
    bool layerEnabled(Layer layer) const override;
    void setLayerEnabled(Layer layer, bool enabled) override;
    Layer addLayer(const std::string& name, LayerRendererType renderer) override;
    bool removeLayer(Layer layer) override;
    Octree* getLayerOctree(Layer layer) override { (void)layer; return nullptr; }
    const Octree* getLayerOctree(Layer layer) const override { (void)layer; return nullptr; }

private:
    struct Stub {
        OctreeNode node;      // address-stable identity for NodeID use
        BoundingCube cube{glm::vec3(0.0f), 0.0f};
        uint32_t level = 0;
        uint64_t serverId = 0;
        uint8_t layer = 0;
        bool alive = true;
    };
    struct CacheKey {
        uint64_t serverId;
        uint32_t version;
        bool operator==(const CacheKey& o) const {
            return serverId == o.serverId && version == o.version;
        }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const noexcept {
            return std::hash<uint64_t>{}(k.serverId * 1000003ull + k.version);
        }
    };
    struct LayerInfo {
        std::string name = "Layer";
        LayerRendererType renderer = LayerRendererType::Solid;
        bool enabled = true;
    };

    void ensureLayerLocked(Layer layer);
    void ensureLayerForMetaLocked(uint8_t layer);
    // Grow layers/meta/handlers to cover `layer` (no locks held on entry;
    // acquires in the global order layers -> meta -> handlers).
    void ensureCapacityForLayer(Layer layer);

    // Registry ----------------------------------------------------------
    Stub* upsertStub(const chunkproto::ChunkRecord& rec); // creates/refreshes
    Stub* findByNode(OctreeNode* node);                   // stubsMutex held
    bool killStub(uint64_t serverId);                     // marks dead, keeps memory

    // Fetch path ----------------------------------------------------------
    // Synchronously requests one mesh (serialized across threads). Returns
    // false on timeout/disconnect/stale reply.
    bool fetchMesh(uint64_t serverId, chunkproto::MeshData& out);
    // Receiver-thread routing for one gunzipped frame.
    void routeFrame(const std::vector<uint8_t>& raw);
    void receiverLoop();
    void disconnectLocked(); // lifeMutex_ held

    // Throttled re-record of a failed chunk so a later dispatch retries.
    void scheduleRetry(Layer layer, uint64_t serverId);

    static Geometry buildGeometry(const chunkproto::MeshData& mesh, int brushIndex);

    // Connection ----------------------------------------------------------
    WsClient client_;
    // Serializes connect/disconnect (each joins the receiver thread).
    std::mutex lifeMutex_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> failed_{false};
    std::thread receiverThread_;

    // Stored per-layer collector lambdas from loadScene() (index = Layer).
    // Guarded by handlersMutex_ because routeFrame runs on the receiver
    // thread while the main thread may grow the layer list.
    mutable std::mutex handlersMutex_;
    std::vector<Octree::OctreeNodeDataHandler> updateHandlers_;
    std::vector<Octree::OctreeNodeDataHandler> deleteHandlers_;

    // Dynamic layer metadata (names/renderers/enabled). Guarded by layersMutex_.
    mutable std::mutex layersMutex_;
    std::vector<LayerInfo> layers_;

    // Stub registry (never frees until disconnect).
    mutable std::mutex stubsMutex_;
    std::unordered_map<uint64_t, std::unique_ptr<Stub>> byServer_;
    std::unordered_map<OctreeNode*, uint64_t> byNode_;
    std::unordered_map<uint64_t, int64_t> lastRetryMs_;

    // Scene meta (root lattice + ladder depth per layer, index = Layer).
    mutable std::mutex metaMutex_;
    std::vector<bool> haveMeta_;
    std::vector<glm::vec3> rootMin_;
    std::vector<float> chunkSize_;
    std::vector<uint8_t> rootChunkLod_;

    // Mesh cache by (serverId, version).
    mutable std::mutex cacheMutex_;
    std::unordered_map<CacheKey, Geometry, CacheKeyHash> meshCache_;

    // Outstanding mesh fetch (at most one per connection; server replies in
    // request order, matched by reply id).
    std::mutex fetchMutex_;
    std::condition_variable fetchCv_;
    bool fetchPending_ = false;
    uint64_t fetchId_ = 0;
    chunkproto::MeshData fetchReply_;
    bool fetchReady_ = false;
    std::mutex sendMutex_; // serializes frame writes across threads
};
