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
#include "server/SceneServer.hpp"
#include "utils/BlenderExport.hpp"
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

// --generate forces the procedural MainSceneLoader scene even when a saved
// scene bundle exists (headless scene/brush testing).
bool parseGenerateArg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--generate") return true;
    }
    return false;
}

// --export-glb <path> / --export-blend <path>: same exporter, the extension
// selects the output (.glb writes the interchange file, .blend additionally
// runs Blender headlessly to save the scene).
std::string parseExportArg(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--export-glb" || a == "--export-blend") && i + 1 < argc) return argv[++i];
        if (a.rfind("--export-glb=", 0) == 0) return a.substr(13);
        if (a.rfind("--export-blend=", 0) == 0) return a.substr(15);
    }
    return {};
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
    const std::string exportGlb = parseExportArg(argc, argv);
    const bool exportMode = !exportGlb.empty();
    const bool forceGenerate = parseGenerateArg(argc, argv);

    LocalScene mainScene;
    // The server class owns the transport + providers; the handlers below feed
    // it live octree edits while the scene is built or edited.
    SceneServer server(mainScene, port, /*verbose=*/true);

    std::atomic<uint64_t> nSolid{0}, nSolidDel{0}, nLiquid{0}, nLiquidDel{0};
    auto makeUpsert = [&](Layer layer) -> Octree::OctreeNodeDataHandler {
        return [&, layer](const OctreeNodeData& nd) {
            if (exportMode) return;
            server.pushUpsert(nd, static_cast<int>(layer));
            if (layer == LAYER_OPAQUE) ++nSolid;
            else if (layer == LAYER_TRANSPARENT) ++nLiquid;
        };
    };
    auto makeErase = [&](Layer layer) -> Octree::OctreeNodeDataHandler {
        return [&, layer](const OctreeNodeData& nd) {
            if (exportMode) return;
            server.pushDelete(nd, static_cast<int>(layer));
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

    if (!exportMode) {
        try {
            server.start();
        } catch (const std::exception& e) {
            std::cerr << "server: web start failed: " << e.what() << std::endl;
            return 1;
        }
        std::cout << "server: viewer at http://localhost:" << port << "/  (ws /ws, health /health)\n"
                  << "server: open the page from the URL above so its WebSocket reaches /ws on the same host:port\n"
                  << "server: (use ?port=<p> or ?ws=ws://host:port/ws to point the page elsewhere)\n";
    }

    // Load AFTER the web server is up so live octree updates stream to
    // browsers that are already connected.
    const std::string scenePath = forceGenerate ? std::string() : findDefaultScene(sceneArg);
    if (forceGenerate) {
        std::cout << "server: --generate: building the procedural MainSceneLoader scene\n";
    }
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
    auto snap = chunksnap::collectAll(server.ids(), mainScene);
    size_t nOpq = 0, nTrn = 0;
    for (auto& r : snap) (r.layer == 0 ? nOpq : nTrn)++;
    std::cout << "server: ready in " << secs << "s"
              << " (chunks=" << snap.size() << " opaque=" << nOpq
              << " transparent=" << nTrn << ")"
              << " upserts(opaque=" << nSolid.load() << " transparent=" << nLiquid.load() << ")"
              << " deletes(opaque=" << nSolidDel.load() << " transparent=" << nLiquidDel.load() << ")\n";

    if (exportMode) {
        blender_export::Options options;
        options.outputPath = exportGlb;
        const blender_export::Stats stats =
            blender_export::exportSceneToBlender(mainScene, options, {},
                [](float f, const std::string& phase) {
                    std::cout << "[server] export " << int(f * 100.0f) << "% " << phase << "\n";
                });
        if (!stats.ok) {
            std::cerr << "server: Blender export failed: " << stats.error << std::endl;
        } else {
            if (stats.blendWritten) {
                std::cout << "server: Blender scene saved: " << stats.blendPath << std::endl;
            }
            if (!stats.warning.empty()) {
                std::cerr << "server: warning: " << stats.warning << std::endl;
            }
            std::cout << "server: Blender export ok: " << stats.outputPath
                      << " script=" << stats.scriptPath
                      << " config=" << stats.configPath << std::endl;
        }
        mainScene.stopPools();
        return stats.ok ? 0 : 1;
    }

    while (g_run) std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "\nserver: shutting down\n";
    server.stop();
    mainScene.stopPools();
    return 0;
}
