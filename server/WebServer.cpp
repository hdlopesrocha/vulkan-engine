#include "WebServer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#include "../utils/WsSha1.hpp"

// ---------------------------------------------------------------- socket helpers ---
namespace {

bool sendAll(int fd, const uint8_t* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}
bool sendAll(int fd, const std::string& s) {
    return sendAll(fd, reinterpret_cast<const uint8_t*>(s.data()), s.size());
}
bool readExact(int fd, uint8_t* out, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = ::recv(fd, out + got, len - got, 0);
        if (n <= 0) return false;
        got += (size_t)n;
    }
    return true;
}

std::string toLower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a]==' '||s[a]=='\t')) ++a;
    while (b > a && (s[b-1]==' '||s[b-1]=='\t'||s[b-1]=='\r')) --b;
    return s.substr(a, b - a);
}
} // namespace

WebServer::WebServer(uint16_t port) : port_(port) {}
WebServer::~WebServer() { stop(); }

void WebServer::setSnapshotProvider(std::function<std::vector<chunkproto::ChunkRecord>()> fn) {
    snapshotProvider_ = std::move(fn);
}

void WebServer::setMetaProvider(std::function<std::vector<chunkproto::SceneMetaLayer>()> fn) {
    metaProvider_ = std::move(fn);
}

void WebServer::setMeshProvider(std::function<bool(uint64_t, chunkproto::MeshData&)> fn) {
    meshProvider_ = std::move(fn);
}

size_t WebServer::clientCount() const {
    std::lock_guard<std::mutex> l(clientsMutex_);
    return clients_.size();
}
uint64_t WebServer::upsertsSent() const { return upsertsSent_.load(); }
uint64_t WebServer::deletesSent() const { return deletesSent_.load(); }
uint64_t WebServer::meshesSent() const { return meshesSent_.load(); }

void WebServer::pushUpsert(const chunkproto::ChunkRecord& rec) {
    std::lock_guard<std::mutex> l(pendingMutex_);
    pendingUpserts_.push_back(rec);
}

void WebServer::pushDelete(uint64_t id) {
    std::lock_guard<std::mutex> l(pendingMutex_);
    pendingDeletes_.push_back(id);
}

std::string WebServer::loadIndexHtml() {
    std::lock_guard<std::mutex> l(indexMutex_);
    // Re-read when the file changed on disk (one stat per page view), so
    // viewer edits take effect without restarting the server.
    if (!indexHtml_.empty() && !indexPath_.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(indexPath_, ec) ||
            std::filesystem::last_write_time(indexPath_, ec) == indexMtime_)
            return indexHtml_;
    }
    static const char* kCandidates[] = {
        "server/web/index.html",     // CWD = repo root
        "../server/web/index.html",  // CWD = bin/
        "web/index.html",            // CWD = bin/ after `make` copy
        "./web/index.html",
        "bin/web/index.html",
    };
    for (const char* p : kCandidates) {
        std::ifstream f(p, std::ios::binary);
        if (!f) continue;
        std::ostringstream ss;
        ss << f.rdbuf();
        indexHtml_ = ss.str();
        indexPath_ = p;
        std::error_code ec;
        indexMtime_ = std::filesystem::last_write_time(indexPath_, ec);
        std::cout << "[web] serving page from '" << p << "' (" << indexHtml_.size() << " bytes)\n";
        return indexHtml_;
    }
    std::cerr << "[web] index.html not found (tried server/web/index.html, ../server/web/index.html, web/index.html)\n";
    return {};
}

void WebServer::start() {
    if (running_) return;
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) throw std::runtime_error("web: socket() failed");
    int one = 1;
    ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port_);
    if (::bind(listenFd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
        int err = errno;
        ::close(listenFd_);
        listenFd_ = -1;
        throw std::runtime_error("web: bind(port " + std::to_string(port_) + ") failed: " +
            std::strerror(err) + " — is another server already holding this port? " +
            "(find it with: ss -ltnp | grep " + std::to_string(port_) + ")");
    }
    if (::listen(listenFd_, 16) != 0) {
        ::close(listenFd_);
        throw std::runtime_error("web: listen() failed");
    }
    running_ = true;
    // Preload page so a missing file is reported at startup, not on first GET.
    loadIndexHtml();
    acceptThread_ = std::thread(&WebServer::acceptLoop, this);
    flusherThread_ = std::thread(&WebServer::flusherLoop, this);
    std::cout << "[web] listening on http://0.0.0.0:" << port_ << "  (page / + ws /ws)\n";
}

void WebServer::stop() {
    if (!running_) return;
    running_ = false;
    if (listenFd_ >= 0) {
        ::shutdown(listenFd_, SHUT_RDWR);
        ::close(listenFd_);
        listenFd_ = -1;
    }
    if (acceptThread_.joinable()) acceptThread_.join();
    if (flusherThread_.joinable()) flusherThread_.join();
    std::lock_guard<std::mutex> l(clientsMutex_);
    for (int fd : clients_) ::close(fd);
    clients_.clear();
}

void WebServer::acceptLoop() {
    while (running_) {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        int fd = ::accept(listenFd_, (sockaddr*)&peer, &plen);
        if (fd < 0) {
            if (running_) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        std::thread(&WebServer::clientThread, this, fd).detach();
    }
}

void WebServer::flusherLoop() {
    using namespace std::chrono;
    while (running_) {
        std::this_thread::sleep_for(milliseconds(50));
        std::vector<chunkproto::ChunkRecord> ups;
        std::vector<uint64_t> dels;
        {
            std::lock_guard<std::mutex> l(pendingMutex_);
            if (pendingUpserts_.empty() && pendingDeletes_.empty()) continue;
            ups.swap(pendingUpserts_);
            dels.swap(pendingDeletes_);
        }
        // Tessellate the upserted chunks BEFORE taking the stream lock (the
        // provider is thread-safe); the MESH_BATCH frames go out with the
        // record batches below under one lock hold so they never interleave
        // inside a concurrent snapshot on another client thread. Empty
        // meshes are omitted (receivers treat a missing id as no geometry).
        std::vector<chunkproto::MeshData> upMeshes;
        if (meshProvider_) {
            upMeshes.reserve(ups.size());
            for (const auto& rec : ups) {
                chunkproto::MeshData mesh;
                mesh.id = rec.id;
                bool ok = false;
                try {
                    ok = meshProvider_(rec.id, mesh);
                } catch (const std::exception& e) {
                    std::cerr << "[web] mesh build failed for id=" << rec.id << ": " << e.what() << "\n";
                    continue;
                }
                if (ok && !(mesh.positions.empty() && mesh.indices.empty()))
                    upMeshes.push_back(std::move(mesh));
            }
        }
        // Send upserts in 512-record batches (each WS frame gzipped alone).
        // The whole flush holds streamMutex_ so it never interleaves inside a
        // concurrent snapshot (START..END) on another client thread.
        std::lock_guard<std::mutex> stream(streamMutex_);
        constexpr size_t kBatch = 512;
        for (size_t i = 0; i < ups.size(); i += kBatch) {
            size_t n = std::min(kBatch, ups.size() - i);
            auto raw = chunkproto::buildChunkBatch(ups.data() + i, n);
            auto gz = chunkproto::gzipCompress(raw);
            broadcastRaw(gz, 0x2);
            upsertsSent_ += n;
        }
        // Geometry for the same upserts, ~512KB raw (or 128 meshes) per frame.
        size_t mo = 0, mBytes = 0, mStart = 0;
        for (size_t i = 0; i < upMeshes.size(); ++i) {
            mo++;
            mBytes += upMeshes[i].positions.size() * 4 + upMeshes[i].indices.size() * 4;
            if (mo >= 128 || mBytes >= (512u << 10)) {
                auto raw = chunkproto::buildMeshBatch(upMeshes.data() + mStart, i + 1 - mStart);
                broadcastRaw(chunkproto::gzipCompress(raw), 0x2);
                mStart = i + 1;
                mo = 0;
                mBytes = 0;
            }
        }
        if (mStart < upMeshes.size()) {
            auto raw = chunkproto::buildMeshBatch(upMeshes.data() + mStart, upMeshes.size() - mStart);
            broadcastRaw(chunkproto::gzipCompress(raw), 0x2);
        }
        for (size_t i = 0; i < dels.size(); i += kBatch) {
            size_t n = std::min(kBatch, dels.size() - i);
            auto raw = chunkproto::buildDeleteBatch(dels.data() + i, n);
            auto gz = chunkproto::gzipCompress(raw);
            broadcastRaw(gz, 0x2);
            deletesSent_ += n;
        }
    }
}

void WebServer::clientThread(int fd) {
    if (!handleHttpOrWs(fd)) ::close(fd);
    // wsLoop closes fd itself on exit.
}

bool WebServer::handleHttpOrWs(int fd) {
    // Read until end of HTTP headers.
    std::string req;
    req.reserve(4096);
    char buf[4096];
    while (req.size() < 65536) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) { ::close(fd); return true; }
        req.append(buf, (size_t)n);
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    size_t eol = req.find("\r\n");
    if (eol == std::string::npos) { ::close(fd); return true; }
    std::istringstream rl(req.substr(0, eol));
    std::string method, target, version;
    rl >> method >> target >> version;
    std::map<std::string, std::string> headers;
    size_t pos = eol + 2;
    while (true) {
        size_t nl = req.find("\r\n", pos);
        if (nl == std::string::npos || nl == pos) break;
        std::string line = req.substr(pos, nl - pos);
        size_t c = line.find(':');
        if (c != std::string::npos)
            headers[toLower(trim(line.substr(0, c)))] = trim(line.substr(c + 1));
        pos = nl + 2;
    }
    // Strip query string.
    size_t q = target.find('?');
    std::string path = (q == std::string::npos) ? target : target.substr(0, q);

    auto itUp = headers.find("upgrade");
    bool wantsWs = (itUp != headers.end() && toLower(itUp->second) == "websocket") &&
                   path == "/ws";
    if (wantsWs) {
        auto itKey = headers.find("sec-websocket-key");
        if (itKey == headers.end()) {
            std::string r = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            sendAll(fd, r);
            ::close(fd);
            return true;
        }
        std::string accept = wssha1::wsAcceptKey(itKey->second);
        std::ostringstream rs;
        rs << "HTTP/1.1 101 Switching Protocols\r\n"
           << "Upgrade: websocket\r\n"
           << "Connection: Upgrade\r\n"
           << "Sec-WebSocket-Accept: " << accept << "\r\n\r\n";
        if (!sendAll(fd, rs.str())) { ::close(fd); return true; }
        {
            std::lock_guard<std::mutex> l(clientsMutex_);
            clients_.push_back(fd);
        }
        std::cout << "[web] ws client connected (fd=" << fd << ", clients=" << clientCount() << ")\n";
        wsLoop(fd); // takes ownership; closes fd
        return true;
    }

    std::string body, ctype = "text/plain";
    int code = 200;
    std::string codeText = "OK";
    if (method != "GET") {
        code = 405; codeText = "Method Not Allowed";
        body = "method not allowed\n";
    } else if (path == "/" || path == "/index.html") {
        body = loadIndexHtml();
        ctype = "text/html; charset=utf-8";
        if (body.empty()) {
            code = 500; codeText = "Internal Server Error";
            body = "index.html not found on server\n";
            ctype = "text/plain";
        }
    } else if (path == "/health") {
        std::ostringstream ss;
        ss << "{\"status\":\"ok\",\"clients\":" << clientCount()
           << ",\"upserts\":" << upsertsSent() << ",\"deletes\":" << deletesSent()
           << ",\"meshes\":" << meshesSent() << "}\n";
        body = ss.str();
        ctype = "application/json";
    } else {
        code = 404; codeText = "Not Found";
        body = "not found\n";
    }
    std::ostringstream rs;
    rs << "HTTP/1.1 " << code << " " << codeText << "\r\n"
       << "Content-Type: " << ctype << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n\r\n";
    sendAll(fd, rs.str());
    sendAll(fd, body);
    ::close(fd);
    return true;
}

void WebServer::sendRaw(int fd, const std::vector<uint8_t>& gzipped, uint8_t opcode) {
    std::lock_guard<std::mutex> l(sendMutex_);
    uint8_t h[10];
    size_t hlen = 0;
    h[0] = (uint8_t)(0x80 | (opcode & 0x0F));
    size_t n = gzipped.size();
    if (n < 126) {
        h[1] = (uint8_t)n; hlen = 2;
    } else if (n < 65536) {
        h[1] = 126; h[2] = (uint8_t)((n >> 8) & 0xFF); h[3] = (uint8_t)(n & 0xFF); hlen = 4;
    } else {
        h[1] = 127;
        for (int i = 0; i < 8; ++i) h[2+i] = (uint8_t)((n >> (56 - 8*i)) & 0xFF);
        hlen = 10;
    }
    // Best-effort: a failed send means the client is gone; wsLoop cleans up.
    sendAll(fd, h, hlen);
    if (n) sendAll(fd, gzipped.data(), n);
}

void WebServer::sendGzipped(int fd, const std::vector<uint8_t>& raw, uint8_t opcode) {
    auto gz = chunkproto::gzipCompress(raw);
    sendRaw(fd, gz, opcode);
}

void WebServer::broadcastRaw(const std::vector<uint8_t>& gzipped, uint8_t opcode) {
    std::vector<int> fds;
    {
        std::lock_guard<std::mutex> l(clientsMutex_);
        fds = clients_;
    }
    for (int fd : fds) sendRaw(fd, gzipped, opcode);
}

void WebServer::sendSnapshotTo(int fd) {
    if (!snapshotProvider_) {
        auto end = chunkproto::buildSnapshotEnd();
        sendGzipped(fd, end, 0x2);
        return;
    }
    std::vector<chunkproto::ChunkRecord> all;
    try {
        all = snapshotProvider_();
    } catch (const std::exception& e) {
        std::cerr << "[web] snapshot failed: " << e.what() << "\n";
        return;
    }
    std::cout << "[web] snapshot -> fd=" << fd << " chunks=" << all.size() << "\n";
    std::lock_guard<std::mutex> stream(streamMutex_);
    sendGzipped(fd, chunkproto::buildSnapshotStart((uint32_t)all.size()), 0x2);
    if (metaProvider_) {
        try {
            auto meta = metaProvider_();
            if (!meta.empty())
                sendGzipped(fd, chunkproto::buildSceneMeta(meta.data(), meta.size()), 0x2);
        } catch (const std::exception& e) {
            std::cerr << "[web] scene-meta failed: " << e.what() << "\n";
        }
    }
    constexpr size_t kBatch = 512;
    for (size_t i = 0; i < all.size(); i += kBatch) {
        size_t n = std::min(kBatch, all.size() - i);
        sendGzipped(fd, chunkproto::buildChunkBatch(all.data() + i, n), 0x2);
    }
    // Geometry push: one MESH_BATCH frame per ~512KB raw (or 128 meshes), so
    // render clients hold tessellation for every chunk without requesting.
    // Empty meshes are omitted; receivers treat a missing id as no geometry.
    if (meshProvider_) {
        std::vector<chunkproto::MeshData> pending;
        size_t pendingBytes = 0, nMesh = 0;
        auto flushMeshes = [&]() {
            if (pending.empty()) return;
            sendGzipped(fd, chunkproto::buildMeshBatch(pending.data(), pending.size()), 0x2);
            nMesh += pending.size();
            pending.clear();
            pendingBytes = 0;
        };
        for (const auto& rec : all) {
            chunkproto::MeshData mesh;
            mesh.id = rec.id;
            bool ok = false;
            try {
                ok = meshProvider_(rec.id, mesh);
            } catch (const std::exception& e) {
                std::cerr << "[web] mesh build failed for id=" << rec.id << ": " << e.what() << "\n";
                continue;
            }
            if (!ok || (mesh.positions.empty() && mesh.indices.empty())) continue;
            pendingBytes += mesh.positions.size() * 4 + mesh.indices.size() * 4;
            pending.push_back(std::move(mesh));
            if (pending.size() >= 128 || pendingBytes >= (512u << 10)) flushMeshes();
        }
        flushMeshes();
        std::cout << "[web] snapshot meshes -> fd=" << fd << " sent=" << nMesh << "\n";
    }
    sendGzipped(fd, chunkproto::buildSnapshotEnd(), 0x2);
}

void WebServer::sendMeshTo(int fd, uint64_t chunkId) {
    chunkproto::MeshData mesh;
    mesh.id = chunkId;
    if (meshProvider_) {
        try {
            meshProvider_(chunkId, mesh);
        } catch (const std::exception& e) {
            std::cerr << "[web] mesh build failed for id=" << chunkId << ": " << e.what() << "\n";
            mesh = chunkproto::MeshData();
            mesh.id = chunkId;
        }
    }
    const size_t nTris = mesh.indices.size() / 3;
    std::cout << "[web] mesh -> fd=" << fd << " id=" << chunkId
              << " verts=" << (mesh.positions.size() / 3)
              << " tris=" << nTris << "\n";
    // Whole unicast holds streamMutex_ so it can never interleave inside a
    // broadcast flush (or snapshot) sharing this fd's stream.
    std::lock_guard<std::mutex> stream(streamMutex_);
    sendGzipped(fd, chunkproto::buildMeshData(mesh), 0x2);
    ++meshesSent_;
}

void WebServer::sendAllMeshesTo(int fd) {
    if (!snapshotProvider_ || !meshProvider_) return;
    std::vector<chunkproto::ChunkRecord> all;
    try {
        all = snapshotProvider_();
    } catch (const std::exception& e) {
        std::cerr << "[web] all-meshes snapshot failed: " << e.what() << "\n";
        return;
    }
    size_t sent = 0, skipped = 0;
    for (const auto& rec : all) {
        chunkproto::MeshData mesh;
        mesh.id = rec.id;
        bool ok = false;
        try {
            ok = meshProvider_(rec.id, mesh);
        } catch (const std::exception& e) {
            std::cerr << "[web] mesh build failed for id=" << rec.id << ": " << e.what() << "\n";
            continue;
        }
        if (!ok || (mesh.positions.empty() && mesh.indices.empty())) { skipped++; continue; }
        // Whole unicast holds streamMutex_ so it can never interleave inside
        // a broadcast flush (or snapshot) sharing this fd's stream.
        std::lock_guard<std::mutex> stream(streamMutex_);
        sendGzipped(fd, chunkproto::buildMeshData(mesh), 0x2);
        ++meshesSent_;
        sent++;
    }
    std::cout << "[web] all-meshes -> fd=" << fd
              << " chunks=" << all.size() << " sent=" << sent
              << " skipped-empty=" << skipped << "\n";
}

void WebServer::wsLoop(int fd) {
    std::vector<uint8_t> frag;
    uint8_t fragOpcode = 0;
    auto removeClient = [&]() {
        std::lock_guard<std::mutex> l(clientsMutex_);
        clients_.erase(std::remove(clients_.begin(), clients_.end(), fd), clients_.end());
    };
    while (running_) {
        uint8_t h[2];
        if (!readExact(fd, h, 2)) break;
        bool fin = (h[0] & 0x80) != 0;
        uint8_t opcode = h[0] & 0x0F;
        bool masked = (h[1] & 0x80) != 0;
        uint64_t len = h[1] & 0x7F;
        if (len == 126) {
            uint8_t e[2];
            if (!readExact(fd, e, 2)) break;
            len = ((uint64_t)e[0] << 8) | e[1];
        } else if (len == 127) {
            uint8_t e[8];
            if (!readExact(fd, e, 8)) break;
            len = 0;
            for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
        }
        if (len > (16u << 20)) break; // 16MB cap per frame
        uint8_t mask[4] = {};
        if (masked && !readExact(fd, mask, 4)) break;
        std::vector<uint8_t> payload((size_t)len);
        if (len && !readExact(fd, payload.data(), (size_t)len)) break;
        if (masked) {
            for (uint64_t i = 0; i < len; ++i) payload[(size_t)i] ^= mask[i % 4];
        }
        if (opcode == 0x8) break; // close
        if (opcode == 0x9) { // ping -> pong
            std::lock_guard<std::mutex> l(sendMutex_);
            uint8_t ph[2] = {(uint8_t)(0x80 | 0xA), (uint8_t)std::min<uint64_t>(len, 125)};
            sendAll(fd, ph, 2);
            if (len) sendAll(fd, payload.data(), std::min<uint64_t>(len, 125));
            continue;
        }
        if (opcode == 0xA) continue; // pong
        if (opcode == 0x0) { // continuation
            frag.insert(frag.end(), payload.begin(), payload.end());
            if (!fin) continue;
            payload.swap(frag);
            frag.clear();
            opcode = fragOpcode;
        } else if (!fin) {
            frag = payload;
            fragOpcode = opcode;
            continue;
        }
        if (opcode == 0x1 || opcode == 0x2) {
            // All client frames are gzip blobs carrying the binary protocol.
            std::vector<uint8_t> raw;
            try {
                raw = chunkproto::gzipDecompress(payload.data(), payload.size());
            } catch (...) {
                // Fall back to raw bytes (lets curl/text probes fail loudly).
                continue;
            }
            if (raw.empty()) continue;
            uint8_t type = raw[0];
            if (type == chunkproto::MSG_REQUEST_ALL) {
                sendSnapshotTo(fd);
            } else if (type == chunkproto::MSG_REQUEST_MESH) {
                // 0x02 + u64 LE chunk id; short/unknown payloads ignored.
                if (raw.size() >= 9) {
                    uint64_t id = 0;
                    for (int i = 0; i < 8; ++i)
                        id |= (uint64_t)raw[1 + i] << (8 * i);
                    sendMeshTo(fd, id);
                }
            } else if (type == chunkproto::MSG_REQUEST_ALL_MESHES) {
                // 0x03: stream one MESH_DATA per live chunk (empties skipped).
                sendAllMeshesTo(fd);
            }
            // (unknown types ignored)
        }
    }
    removeClient();
    ::close(fd);
    std::cout << "[web] ws client disconnected (fd=" << fd << ", clients=" << clientCount() << ")\n";
}
