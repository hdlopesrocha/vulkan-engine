#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "server/ChunkIds.hpp"
#include "server/ChunkMesh.hpp"
#include "server/ChunkSnapshot.hpp"
#include "server/WebServer.hpp"
#include "utils/LocalScene.hpp"
#include "utils/MainSceneLoader.hpp"

namespace {
std::atomic<bool> g_run{true};
void onSignal(int) { g_run = false; }

uint16_t parsePort(int argc, char** argv) {
    uint16_t port = 8080;
    const char* env = std::getenv("CHUNK_PORT");
    if (env && *env) {
        try { port = static_cast<uint16_t>(std::stoi(env)); } catch (...) {}
    }
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--port" || a == "-p") && i + 1 < argc) {
            try { port = static_cast<uint16_t>(std::stoi(argv[++i])); } catch (...) {}
        } else if (a.rfind("--port=", 0) == 0) {
            try { port = static_cast<uint16_t>(std::stoi(a.substr(7))); } catch (...) {}
        }
    }
    return port;
}

std::string parseSceneArg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc) return argv[++i];
        if (a.rfind("--scene=", 0) == 0) return a.substr(8);
    }
    return {};
}

std::string findDefaultScene(const std::string& overridePath) {
    if (!overridePath.empty()) return overridePath;
    static const char* kCandidates[] = {
        "scenes/default.scene",       // CWD = bin/
        "../scenes/default.scene",    // CWD = bin/, source tree scenes
        "bin/scenes/default.scene",   // CWD = repo root
        "scenes/default.scene",
    };
    for (const char* p : kCandidates) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) return p;
    }
    return {};
}
} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    const uint16_t port = parsePort(argc, argv);
    const std::string sceneArg = parseSceneArg(argc, argv);

    LocalScene mainScene;
    WebServer web(port);

    // Opaque chunk ids: the wire protocol must never carry node pointers
    // (see server/ChunkIds.hpp). One registry for the scene's lifetime.
    ChunkIdRegistry chunkIds;

    // Snapshot provider: full chunk listing for REQUEST_ALL.
    web.setSnapshotProvider([&mainScene, &chunkIds]() {
        return chunksnap::collectAll(chunkIds, mainScene);
    });

    // Scene-meta provider: root lattice + ladder depth per layer.
    web.setMetaProvider([&mainScene]() {
        return chunksnap::collectMeta(mainScene);
    });

    // Mesh provider: tessellate one chunk for REQUEST_MESH (wireframe overlay).
    web.setMeshProvider([&mainScene, &chunkIds](uint64_t id, chunkproto::MeshData& mesh) {
        return chunkmesh::buildChunkMesh(chunkIds, mainScene, id, mesh);
    });

    std::atomic<uint64_t> nSolid{0}, nSolidDel{0}, nLiquid{0}, nLiquidDel{0};

    // Dynamic layers: one upsert/erase pair per scene layer (index = Layer).
    // Layers 0/1 keep the historic opaque/transparent counters.
    auto makeUpsert = [&](Layer layer) -> Octree::OctreeNodeDataHandler {
        return [&, layer](const OctreeNodeData& nd) {
            web.pushUpsert(chunksnap::makeRecord(chunkIds, nd, static_cast<uint8_t>(layer)));
            if (layer == LAYER_OPAQUE) {
                if ((++nSolid % 2000) == 0)
                    std::cout << "[server] opaque upserts=" << nSolid.load() << "\n";
            } else if (layer == LAYER_TRANSPARENT) {
                if ((++nLiquid % 2000) == 0)
                    std::cout << "[server] transparent upserts=" << nLiquid.load() << "\n";
            }
        };
    };
    auto makeErase = [&](Layer layer) -> Octree::OctreeNodeDataHandler {
        return [&, layer](const OctreeNodeData& nd) {
            const uint64_t id = chunkIds.remove(nd.node);
            if (id != ChunkIdRegistry::kInvalid) web.pushDelete(id);
            if (layer == LAYER_OPAQUE) ++nSolidDel;
            else if (layer == LAYER_TRANSPARENT) ++nLiquidDel;
        };
    };
    const size_t nLayers = mainScene.layerCount();
    std::vector<Octree::OctreeNodeDataHandler> upserts;
    std::vector<Octree::OctreeNodeDataHandler> erases;
    upserts.reserve(nLayers);
    erases.reserve(nLayers);
    for (size_t i = 0; i < nLayers; ++i) {
        upserts.push_back(makeUpsert(static_cast<Layer>(i)));
        erases.push_back(makeErase(static_cast<Layer>(i)));
    }

    try {
        web.start();
    } catch (const std::exception& e) {
        std::cerr << "server: web start failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "server: viewer at http://localhost:" << port << "/  (ws /ws, health /health)\n"
              << "server: open the page from the URL above so its WebSocket reaches /ws on the same host:port\n"
              << "server: (use ?port=<p> or ?ws=ws://host:port/ws to point the page elsewhere)\n";

    // Load AFTER the web server is up so live octree updates stream to
    // browsers that are already connected.
    const std::string scenePath = findDefaultScene(sceneArg);
    auto t0 = std::chrono::steady_clock::now();
    if (!scenePath.empty()) {
        std::cout << "server: loading scene file '" << scenePath << "'\n";
        mainScene.load(scenePath, upserts, erases, nullptr);
    } else {
        std::cout << "server: no scene file found, generating procedural map\n";
        MainSceneLoader mainSceneLoader;
        mainScene.loadScene(mainSceneLoader, upserts, erases);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    auto snap = chunksnap::collectAll(chunkIds, mainScene);
    size_t nOpq = 0, nTrn = 0;
    for (auto& r : snap) (r.layer == 0 ? nOpq : nTrn)++;
    std::cout << "server: ready in " << secs << "s"
              << " (chunks=" << snap.size() << " opaque=" << nOpq
              << " transparent=" << nTrn << ")"
              << " upserts(opaque=" << nSolid.load() << " transparent=" << nLiquid.load() << ")"
              << " deletes(opaque=" << nSolidDel.load() << " transparent=" << nLiquidDel.load() << ")\n";

    while (g_run) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "\nserver: shutting down\n";
    web.stop();
    mainScene.stopPools();
    return 0;
}
