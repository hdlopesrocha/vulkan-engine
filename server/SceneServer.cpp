#include "SceneServer.hpp"

#include <iostream>
#include <stdexcept>

#include "ChunkMesh.hpp"
#include "ChunkSnapshot.hpp"
#include "../space/OctreeNodeData.hpp"
#include "../utils/LocalScene.hpp"

SceneServer::SceneServer(LocalScene& scene, uint16_t port, bool verbose)
    : scene_(scene), web_(port), verbose_(verbose) {}

SceneServer::~SceneServer() {
    stop();
}

void SceneServer::start() {
    if (running_.load()) return;
    error_.clear();

    // Full chunk listing + scene metadata for REQUEST_ALL. Both take the
    // scene's tree lock internally, so this is safe while the application
    // keeps editing the octree.
    web_.setSnapshotProvider([this]() {
        return chunksnap::collectAll(ids_, scene_);
    });
    web_.setMetaProvider([this]() {
        return chunksnap::collectMeta(scene_);
    });
    // One chunk's tessellated mesh (wireframe overlay / REQUEST_MESH).
    web_.setMeshProvider([this](uint64_t id, chunkproto::MeshData& mesh) {
        return chunkmesh::buildChunkMesh(ids_, scene_, id, mesh);
    });

    try {
        web_.start();
    } catch (const std::exception& e) {
        error_ = e.what();
        throw;
    }
    running_.store(true);
    if (verbose_) {
        std::cout << "[SceneServer] serving on http://localhost:" << port()
                  << "/ (ws /ws, health /health), layers=" << scene_.layerCount() << std::endl;
    }
}

void SceneServer::stop() {
    if (!running_.exchange(false)) return;
    web_.stop();
    if (verbose_) {
        const Counters c = counters();
        std::cout << "[SceneServer] stopped (upserts=" << c.upserts << " deletes=" << c.deletes
                  << " clients=" << web_.clientCount() << ")" << std::endl;
    }
}

void SceneServer::pushUpsert(const OctreeNodeData& nd, int layer) {
    if (!running_.load()) return;
    web_.pushUpsert(chunksnap::makeRecord(ids_, nd, static_cast<uint8_t>(layer)));
    upserts_.fetch_add(1);
    if (layer == 0) {
        if ((solidUpserts_.fetch_add(1) + 1) % 2000 == 0 && verbose_) {
            std::cout << "[SceneServer] opaque upserts=" << solidUpserts_.load() << std::endl;
        }
    } else {
        if ((transparentUpserts_.fetch_add(1) + 1) % 2000 == 0 && verbose_) {
            std::cout << "[SceneServer] transparent upserts=" << transparentUpserts_.load() << std::endl;
        }
    }
}

void SceneServer::pushDelete(const OctreeNodeData& nd, int layer) {
    if (!running_.load()) return;
    const uint64_t id = ids_.remove(nd.node);
    if (id == ChunkIdRegistry::kInvalid) return;
    web_.pushDelete(id);
    deletes_.fetch_add(1);
    if (layer == 0) {
        solidDeletes_.fetch_add(1);
    } else {
        transparentDeletes_.fetch_add(1);
    }
}

SceneServer::Counters SceneServer::counters() const {
    Counters c;
    c.upserts = upserts_.load();
    c.deletes = deletes_.load();
    c.solidUpserts = solidUpserts_.load();
    c.transparentUpserts = transparentUpserts_.load();
    c.solidDeletes = solidDeletes_.load();
    c.transparentDeletes = transparentDeletes_.load();
    return c;
}
