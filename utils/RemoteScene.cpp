#include "RemoteScene.hpp"

// NOTE: the wire codec lives in server/ (compiled into the app too, see
// Makefile SRCS). The protocol itself is client/server symmetric: every
// frame is gzip + the binary layout documented in ChunkProtocol.hpp.
#include "../server/ChunkProtocol.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool checkMagic(const std::vector<uint8_t>& raw) {
    return raw.size() >= 5 && raw[0] == 'C' && raw[1] == 'H' && raw[2] == 'K' && raw[3] == '1';
}

uint64_t readU64LE(const uint8_t* p) {
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i);
    return x;
}
uint32_t readU32LE(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

} // namespace

RemoteScene::RemoteScene() {
    std::lock_guard<std::mutex> lock(layersMutex_);
    layers_.reserve(2);
    layers_.push_back({"Opaque", LayerRendererType::Solid, true});
    layers_.push_back({"Transparent", LayerRendererType::Water, true});
    {
        std::lock_guard<std::mutex> mlock(metaMutex_);
        haveMeta_.assign(2, false);
        rootMin_.assign(2, glm::vec3(0.0f));
        chunkSize_.assign(2, 0.0f);
        rootChunkLod_.assign(2, 0);
    }
}

RemoteScene::~RemoteScene() {
    disconnect();
}

size_t RemoteScene::layerCount() const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    return layers_.size();
}

std::string RemoteScene::layerName(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return {};
    return layers_[static_cast<size_t>(layer)].name;
}

void RemoteScene::setLayerName(Layer layer, const std::string& name) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].name = name;
}

LayerRendererType RemoteScene::layerRenderer(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size())
        return (layer == LAYER_TRANSPARENT) ? LayerRendererType::Water : LayerRendererType::Solid;
    return layers_[static_cast<size_t>(layer)].renderer;
}

void RemoteScene::setLayerRenderer(Layer layer, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].renderer = renderer;
}

bool RemoteScene::layerEnabled(Layer layer) const {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    return layers_[static_cast<size_t>(layer)].enabled;
}

void RemoteScene::setLayerEnabled(Layer layer, bool enabled) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return;
    layers_[static_cast<size_t>(layer)].enabled = enabled;
}

Layer RemoteScene::addLayer(const std::string& name, LayerRendererType renderer) {
    std::lock_guard<std::mutex> lock(layersMutex_);
    LayerInfo info;
    info.name = name.empty() ? ("Layer " + std::to_string(layers_.size())) : name;
    info.renderer = renderer;
    info.enabled = true;
    layers_.push_back(info);
    const size_t n = layers_.size();
    {
        std::lock_guard<std::mutex> mlock(metaMutex_);
        if (haveMeta_.size() < n) {
            haveMeta_.resize(n, false);
            rootMin_.resize(n, glm::vec3(0.0f));
            chunkSize_.resize(n, 0.0f);
            rootChunkLod_.resize(n, 0);
        }
    }
    {
        std::lock_guard<std::mutex> hlock(handlersMutex_);
        if (updateHandlers_.size() < n) updateHandlers_.resize(n);
        if (deleteHandlers_.size() < n) deleteHandlers_.resize(n);
    }
    return static_cast<Layer>(n - 1);
}

bool RemoteScene::removeLayer(Layer layer) {
    if (layer < 0) return false;
    std::lock_guard<std::mutex> lock(layersMutex_);
    if (static_cast<size_t>(layer) >= layers_.size() || layers_.size() <= 1) return false;
    layers_.erase(layers_.begin() + layer);
    return true;
}

void RemoteScene::ensureLayerLocked(Layer layer) {
    // layersMutex_ held by caller.
    if (layer < 0) return;
    const size_t need = static_cast<size_t>(layer) + 1;
    if (layers_.size() >= need) return;
    const size_t from = layers_.size();
    layers_.reserve(need);
    for (size_t i = from; i < need; ++i) {
        LayerInfo info;
        info.name = "Layer " + std::to_string(i);
        info.renderer = (static_cast<Layer>(i) == LAYER_TRANSPARENT)
            ? LayerRendererType::Water : LayerRendererType::Solid;
        info.enabled = true;
        layers_.push_back(info);
    }
}

void RemoteScene::ensureLayerForMetaLocked(uint8_t layer) {
    // Caller holds metaMutex_. Only grows the meta vectors here; layer +
    // handler growth is done via ensureCapacityForLayer (no locks held) to
    // keep a single global lock order (layers -> meta -> handlers).
    const size_t need = static_cast<size_t>(layer) + 1;
    if (haveMeta_.size() < need) {
        haveMeta_.resize(need, false);
        rootMin_.resize(need, glm::vec3(0.0f));
        chunkSize_.resize(need, 0.0f);
        rootChunkLod_.resize(need, 0);
    }
}

void RemoteScene::ensureCapacityForLayer(Layer layer) {
    if (layer < 0) return;
    const size_t need = static_cast<size_t>(layer) + 1;
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        ensureLayerLocked(layer);
    }
    {
        std::lock_guard<std::mutex> mlock(metaMutex_);
        ensureLayerForMetaLocked(static_cast<uint8_t>(layer));
    }
    {
        std::lock_guard<std::mutex> hlock(handlersMutex_);
        if (updateHandlers_.size() < need) updateHandlers_.resize(need);
        if (deleteHandlers_.size() < need) deleteHandlers_.resize(need);
    }
}

bool RemoteScene::connect(const Endpoint& ep, int timeoutMs) {
    std::lock_guard<std::mutex> life(lifeMutex_);
    disconnectLocked();
    failed_ = false;
    std::cout << "[RemoteScene] connecting to " << ep.host << ":" << ep.port << " ...\n";
    if (!client_.connect(ep.host, ep.port, "/ws", timeoutMs)) {
        std::cout << "[RemoteScene] unreachable (" << client_.lastError()
                  << "); keeping LocalScene\n";
        return false;
    }
    connected_ = true;
    std::cout << "[RemoteScene] WebSocket handshake ok\n";
    return true;
}

void RemoteScene::disconnect() {
    std::lock_guard<std::mutex> life(lifeMutex_);
    disconnectLocked();
}

void RemoteScene::disconnectLocked() {
    connected_ = false;
    client_.close();
    if (receiverThread_.joinable() &&
        receiverThread_.get_id() != std::this_thread::get_id()) {
        receiverThread_.join();
    }
}

// ---------------------------------------------------------------- registry

RemoteScene::Stub* RemoteScene::upsertStub(const chunkproto::ChunkRecord& rec) {
    ensureCapacityForLayer(static_cast<Layer>(rec.layer));
    std::lock_guard<std::mutex> lock(stubsMutex_);
    auto it = byServer_.find(rec.id);
    Stub* stub = nullptr;
    if (it == byServer_.end()) {
        auto owned = std::make_unique<Stub>();
        stub = owned.get();
        byServer_[rec.id] = std::move(owned);
        byNode_[&stub->node] = rec.id;
    } else {
        stub = it->second.get();
    }
    // Refresh all published attributes (same id always names the same node
    // while alive — server ids are never recycled, see ChunkIds.hpp).
    stub->serverId = rec.id;
    stub->layer = rec.layer;
    stub->level = rec.level;
    stub->cube = BoundingCube(glm::vec3(rec.minX, rec.minY, rec.minZ), rec.size);
    stub->alive = true;
    stub->node.setType(static_cast<SpaceType>(rec.spaceType));
    stub->node.setLod(rec.lod);
    stub->node.setChunkLod(rec.chunkLod);
    stub->node.setChunk(true);
    stub->node.setBrush(rec.brushIndex);
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rec.sdf[i];
    stub->node.setSDF(sdf);
    stub->node.version = rec.version;
    return stub;
}

RemoteScene::Stub* RemoteScene::findByNode(OctreeNode* node) {
    auto it = byNode_.find(node);
    if (it == byNode_.end()) return nullptr;
    auto jt = byServer_.find(it->second);
    return jt != byServer_.end() ? jt->second.get() : nullptr;
}

// ---------------------------------------------------------------- snapshot

void RemoteScene::loadScene(SceneLoaderCallback& /*callback*/,
                            std::vector<Octree::OctreeNodeDataHandler> updateHandlers,
                            std::vector<Octree::OctreeNodeDataHandler> deleteHandlers) {
    {
        std::lock_guard<std::mutex> hlock(handlersMutex_);
        updateHandlers_ = std::move(updateHandlers);
        deleteHandlers_ = std::move(deleteHandlers);
        // Guarantee at least the two default layers so old two-handler callers
        // (now vector callers) keep working.
        if (updateHandlers_.size() < 2) updateHandlers_.resize(2);
        if (deleteHandlers_.size() < 2) deleteHandlers_.resize(2);
    }
    {
        std::lock_guard<std::mutex> lock(layersMutex_);
        if (layers_.size() < 2) {
            while (layers_.size() < 2) {
                const size_t i = layers_.size();
                layers_.push_back({i == 1 ? "Transparent" : "Opaque",
                    i == 1 ? LayerRendererType::Water : LayerRendererType::Solid, true});
            }
        }
        const size_t n = std::max(updateHandlers_.size(), layers_.size());
        if (layers_.size() < n) {
            for (size_t i = layers_.size(); i < n; ++i)
                layers_.push_back({"Layer " + std::to_string(i), LayerRendererType::Solid, true});
        }
    }
    {
        std::lock_guard<std::mutex> mlock(metaMutex_);
        const size_t n = std::max(updateHandlers_.size(), layers_.size());
        if (haveMeta_.size() < n) {
            haveMeta_.resize(n, false);
            rootMin_.resize(n, glm::vec3(0.0f));
            chunkSize_.resize(n, 0.0f);
            rootChunkLod_.resize(n, 0);
        }
    }

    if (!connected_) {
        failed_ = true;
        return;
    }
    std::cout << "[RemoteScene] requesting snapshot ...\n";
    {
        std::lock_guard<std::mutex> lock(sendMutex_);
        auto gz = chunkproto::gzipCompress(chunkproto::buildRequestAll());
        if (!client_.sendBinary(gz)) {
            std::cout << "[RemoteScene] snapshot request failed (" << client_.lastError() << ")\n";
            connected_ = false;
            failed_ = true;
            return;
        }
    }

    // Synchronous snapshot loop (mirrors the local blocking octree build):
    // START/META/BATCH.../END, replaying records into the stored handlers.
    // The receiver thread starts only after END so no frame can bypass the
    // replay order.
    size_t nChunks = 0;
    const auto t0 = nowMs();
    for (;;) {
        std::vector<uint8_t> frame;
        if (!client_.recvMessage(frame, 30000)) {
            std::cout << "[RemoteScene] snapshot interrupted (" << client_.lastError() << ")\n";
            connected_ = false;
            failed_ = true;
            return;
        }
        std::vector<uint8_t> raw;
        try {
            raw = chunkproto::gzipDecompress(frame);
        } catch (const std::exception& e) {
            std::cout << "[RemoteScene] dropping corrupt snapshot frame (" << e.what() << ")\n";
            continue;
        }
        if (!checkMagic(raw)) continue;
        const uint8_t type = raw[4];
        if (type == chunkproto::MSG_SNAPSHOT_START) {
            const uint32_t total = raw.size() >= 9 ? readU32LE(raw.data() + 5) : 0;
            std::cout << "[RemoteScene] snapshot start: expect " << total << " chunks\n";
        } else if (type == chunkproto::MSG_SCENE_META) {
            if (raw.size() >= 9) {
                const uint32_t n = readU32LE(raw.data() + 5);
                size_t off = 9;
                int maxLayer = -1;
                {
                    std::lock_guard<std::mutex> lock(metaMutex_);
                    for (uint32_t i = 0; i < n && off + 24 <= raw.size(); ++i, off += 24) {
                        const uint8_t layer = raw[off];
                        ensureLayerForMetaLocked(layer);
                        memcpy(&rootMin_[layer], raw.data() + off + 1, 12);
                        float rootLen, chunkSize;
                        memcpy(&rootLen, raw.data() + off + 13, 4);
                        memcpy(&chunkSize, raw.data() + off + 17, 4);
                        (void)rootLen;
                        chunkSize_[layer] = chunkSize;
                        rootChunkLod_[layer] = raw[off + 21];
                        haveMeta_[layer] = true;
                        maxLayer = std::max<int>(maxLayer, layer);
                    }
                }
                if (maxLayer >= 0) ensureCapacityForLayer(maxLayer);
            }
        } else if (type == chunkproto::MSG_CHUNK_BATCH) {
            if (raw.size() < 7) continue;
            const uint16_t n = (uint16_t)(raw[5] | (raw[6] << 8));
            for (uint16_t i = 0; i < n; ++i) {
                const size_t off = 7 + (size_t)i * chunkproto::CHUNK_RECORD_SIZE;
                if (off + chunkproto::CHUNK_RECORD_SIZE > raw.size()) break;
                chunkproto::ChunkRecord rec;
                if (!chunkproto::decodeRecord(raw.data() + off, chunkproto::CHUNK_RECORD_SIZE, rec))
                    continue;
                Stub* stub = upsertStub(rec);
                if (!stub) continue;
                OctreeNodeData nd(stub->level, &stub->node, stub->cube, nullptr);
                Octree::OctreeNodeDataHandler handler;
                {
                    std::lock_guard<std::mutex> hlock(handlersMutex_);
                    if (stub->layer < updateHandlers_.size()) handler = updateHandlers_[stub->layer];
                }
                if (handler) handler(nd);
                ++nChunks;
            }
        } else if (type == chunkproto::MSG_DELETE_BATCH) {
            if (raw.size() < 7) continue;
            const uint16_t n = (uint16_t)(raw[5] | (raw[6] << 8));
            for (uint16_t i = 0; i < n; ++i) {
                const size_t off = 7 + (size_t)i * 8;
                if (off + 8 > raw.size()) break;
                const uint64_t id = readU64LE(raw.data() + off);
                std::lock_guard<std::mutex> lock(stubsMutex_);
                auto it = byServer_.find(id);
                if (it == byServer_.end()) continue;
                Stub* stub = it->second.get();
                const uint8_t layer = stub->layer;
                stub->alive = false;
                OctreeNodeData nd(stub->level, &stub->node, stub->cube, nullptr);
                Octree::OctreeNodeDataHandler handler;
                {
                    std::lock_guard<std::mutex> hlock(handlersMutex_);
                    if (layer < deleteHandlers_.size()) handler = deleteHandlers_[layer];
                }
                // Collector lambdas only touch their own map: safe to call
                // with stubsMutex held (no stub access inside).
                if (handler) handler(nd);
            }
        } else if (type == chunkproto::MSG_SNAPSHOT_END) {
            break;
        }
    }
    std::cout << "[RemoteScene] snapshot ok: " << nChunks << " chunks in "
              << (nowMs() - t0) << " ms\n";
    receiverThread_ = std::thread(&RemoteScene::receiverLoop, this);
}

// ---------------------------------------------------------------- live Rx

void RemoteScene::routeFrame(const std::vector<uint8_t>& raw) {
    const uint8_t type = raw[4];
    if (type == chunkproto::MSG_CHUNK_BATCH) {
        if (raw.size() < 7) return;
        const uint16_t n = (uint16_t)(raw[5] | (raw[6] << 8));
        for (uint16_t i = 0; i < n; ++i) {
            const size_t off = 7 + (size_t)i * chunkproto::CHUNK_RECORD_SIZE;
            if (off + chunkproto::CHUNK_RECORD_SIZE > raw.size()) break;
            chunkproto::ChunkRecord rec;
            if (!chunkproto::decodeRecord(raw.data() + off, chunkproto::CHUNK_RECORD_SIZE, rec))
                continue;
            Stub* stub = upsertStub(rec);
            if (!stub) continue;
            OctreeNodeData nd(stub->level, &stub->node, stub->cube, nullptr);
            Octree::OctreeNodeDataHandler handler;
            {
                std::lock_guard<std::mutex> hlock(handlersMutex_);
                if (stub->layer < updateHandlers_.size()) handler = updateHandlers_[stub->layer];
            }
            if (handler) handler(nd);
        }
    } else if (type == chunkproto::MSG_DELETE_BATCH) {
        if (raw.size() < 7) return;
        const uint16_t n = (uint16_t)(raw[5] | (raw[6] << 8));
        for (uint16_t i = 0; i < n; ++i) {
            const size_t off = 7 + (size_t)i * 8;
            if (off + 8 > raw.size()) break;
            const uint64_t id = readU64LE(raw.data() + off);
            std::lock_guard<std::mutex> lock(stubsMutex_);
            auto it = byServer_.find(id);
            if (it == byServer_.end()) continue;
            Stub* stub = it->second.get();
            const uint8_t layer = stub->layer;
            stub->alive = false;
            OctreeNodeData nd(stub->level, &stub->node, stub->cube, nullptr);
            Octree::OctreeNodeDataHandler handler;
            {
                std::lock_guard<std::mutex> hlock(handlersMutex_);
                if (layer < deleteHandlers_.size()) handler = deleteHandlers_[layer];
            }
            if (handler) handler(nd);
        }
    } else if (type == chunkproto::MSG_SCENE_META) {
        // Scene reload server-side: refresh the gate inputs (same layout as
        // the snapshot path).
        if (raw.size() < 9) return;
        const uint32_t n = readU32LE(raw.data() + 5);
        size_t off = 9;
        int maxLayer = -1;
        {
            std::lock_guard<std::mutex> lock(metaMutex_);
            for (uint32_t i = 0; i < n && off + 24 <= raw.size(); ++i, off += 24) {
                const uint8_t layer = raw[off];
                ensureLayerForMetaLocked(layer);
                memcpy(&rootMin_[layer], raw.data() + off + 1, 12);
                float chunkSize;
                memcpy(&chunkSize, raw.data() + off + 17, 4);
                chunkSize_[layer] = chunkSize;
                rootChunkLod_[layer] = raw[off + 21];
                haveMeta_[layer] = true;
                maxLayer = std::max<int>(maxLayer, layer);
            }
        }
        if (maxLayer >= 0) ensureCapacityForLayer(maxLayer);
    }
    // SNAPSHOT_START/END outside a snapshot: ignore.
}

void RemoteScene::receiverLoop() {
    while (connected_) {
        std::vector<uint8_t> frame;
        if (!client_.recvMessage(frame)) break; // closed/timeout -> dead
        std::vector<uint8_t> raw;
        try {
            raw = chunkproto::gzipDecompress(frame);
        } catch (const std::exception& e) {
            std::cout << "[RemoteScene] dropping corrupt live frame (" << e.what() << ")\n";
            continue;
        }
        if (!checkMagic(raw)) continue;
        routeFrame(raw);
    }
    if (connected_) {
        std::cout << "[RemoteScene] connection lost (" << client_.lastError()
                  << "); rendering frozen at last state\n";
    }
    connected_ = false;
}

void RemoteScene::reemitLayer(Layer layer, const Octree::OctreeNodeDataHandler& updateHandler) {
    if (!updateHandler || layer < 0) return;
    // Snapshot the layer's live stubs under lock, then invoke the handler
    // without holding stubsMutex_ (collector lambdas only touch their own
    // map; stub addresses are stable until disconnect).
    std::vector<OctreeNodeData> pending;
    {
        std::lock_guard<std::mutex> lock(stubsMutex_);
        pending.reserve(byServer_.size());
        for (const auto& kv : byServer_) {
            const Stub* stub = kv.second.get();
            if (!stub || !stub->alive || stub->layer != static_cast<uint8_t>(layer)) continue;
            pending.emplace_back(stub->level, const_cast<OctreeNode*>(&stub->node), stub->cube, nullptr);
        }
    }
    for (const auto& nd : pending) updateHandler(nd);
}

// ---------------------------------------------------------------- Scene

void RemoteScene::requestSDFCubes(Layer layer, OctreeNodeData& data,
                                  const SdfCubeCallback& callback, ThreadPool* /*poolOverride*/) {
    (void)layer;
    if (!connected_ || !data.node || !callback) return;
    std::lock_guard<std::mutex> lock(stubsMutex_);
    Stub* stub = findByNode(data.node);
    if (!stub || !stub->alive) return;
    // One debug cube per chunk (chunk-level attributes) instead of the local
    // per-node lod==1 walk; the debug SDF renderer consumes them unchanged.
    std::array<float, 8> sdf;
    for (int i = 0; i < 8; ++i) sdf[i] = stub->node.sdf[i];
    callback(stub->cube, sdf, 1u, stub->node.version,
             reinterpret_cast<uintptr_t>(&stub->node),
             static_cast<uint32_t>(stub->node.getBrush()));
}

void RemoteScene::requestBoundingBoxes(Layer layer, OctreeNodeData& data,
                                       const BBoxCallback& callback,
                                       ThreadPool* /*poolOverride*/) {
    (void)layer;
    if (!connected_ || !data.node || !callback) return;
    std::lock_guard<std::mutex> lock(stubsMutex_);
    Stub* stub = findByNode(data.node);
    if (!stub || !stub->alive) return;
    // Chunk-level box instead of the local per-node walk; same consumer.
    callback(stub->cube);
}

bool RemoteScene::isNodeUpToDate(Layer layer, OctreeNodeData& data, uint version) {
    (void)layer;
    if (!data.node) return false;
    std::lock_guard<std::mutex> lock(stubsMutex_);
    Stub* stub = findByNode(data.node);
    return stub && stub->alive && stub->node.version >= version;
}

int RemoteScene::maxChunkLod(Layer layer, float minSize) const {
    // Mirrors LocalScene::maxChunkLod with the META-provided ingredients:
    // min(heightRootToChunk(0, minSize), rootChunkLod - 1), floored at 0.
    std::lock_guard<std::mutex> lock(metaMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= haveMeta_.size()) return 0;
    const size_t li = static_cast<size_t>(layer);
    if (!haveMeta_[li] || chunkSize_[li] <= 0.0f || rootChunkLod_[li] == 0) return 0;
    const float ratio = std::max(1.0f, chunkSize_[li] / std::max(minSize, 1.0f));
    const int maxLevels = static_cast<int>(std::floor(std::log2(ratio)));
    return std::max(0, std::min(maxLevels, static_cast<int>(rootChunkLod_[li]) - 1));
}

glm::vec3 RemoteScene::lodRootMin(Layer layer) const {
    std::lock_guard<std::mutex> lock(metaMutex_);
    if (layer < 0 || static_cast<size_t>(layer) >= rootMin_.size()) return glm::vec3(0.0f);
    return rootMin_[static_cast<size_t>(layer)];
}
