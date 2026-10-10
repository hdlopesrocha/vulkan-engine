#include "WsClient.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>

// ---------------------------------------------------------------- SHA1 ---
// Minimal public-domain SHA1 for the handshake accept-key check. Mirrors the
// implementation in server/WebServer.cpp (kept as a local copy so utils/
// does not depend on server/; both sides must agree on the RFC 6455 key
// transform, and the unit test pins it against the RFC vector).
namespace {
struct Sha1 {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint64_t bitLen = 0;
    uint8_t buf[64] = {};
    size_t bufLen = 0;
    static uint32_t rol(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }
    void block(const uint8_t* p) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) |
                   ((uint32_t)p[4*i+2] << 8) | (uint32_t)p[4*i+3];
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f,k;
            if (i < 20)      { f=(b&c)|((~b)&d); k=0x5A827999; }
            else if (i < 40) { f=b^c^d;          k=0x6ED9EBA1; }
            else if (i < 60) { f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
            else             { f=b^c^d;          k=0xCA62C1D6; }
            uint32_t t = rol(a,5)+f+e+k+w[i];
            e=d; d=c; c=rol(b,30); b=a; a=t;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e;
    }
    void update(const uint8_t* data, size_t len) {
        bitLen += (uint64_t)len * 8;
        while (len) {
            size_t take = std::min(len, 64 - bufLen);
            memcpy(buf + bufLen, data, take);
            bufLen += take; data += take; len -= take;
            if (bufLen == 64) { block(buf); bufLen = 0; }
        }
    }
    void final(uint8_t out[20]) {
        uint8_t pad = 0x80, zero = 0x00;
        update(&pad, 1);
        while (bufLen != 56) update(&zero, 1);
        uint8_t lenBytes[8];
        for (int i = 0; i < 8; ++i) lenBytes[i] = (uint8_t)((bitLen >> (56 - 8*i)) & 0xFF);
        memcpy(buf + bufLen, lenBytes, 8);
        block(buf);
        for (int i = 0; i < 5; ++i) {
            out[4*i]   = (uint8_t)((h[i] >> 24) & 0xFF);
            out[4*i+1] = (uint8_t)((h[i] >> 16) & 0xFF);
            out[4*i+2] = (uint8_t)((h[i] >> 8) & 0xFF);
            out[4*i+3] = (uint8_t)(h[i] & 0xFF);
        }
    }
};

std::string base64Encode(const uint8_t* data, size_t len) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string s;
    s.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        int rem = (int)(len - i);
        if (rem > 1) n |= (uint32_t)data[i+1] << 8;
        if (rem > 2) n |= data[i+2];
        s.push_back(T[(n >> 18) & 63]);
        s.push_back(T[(n >> 12) & 63]);
        s.push_back(rem > 1 ? T[(n >> 6) & 63] : '=');
        s.push_back(rem > 2 ? T[n & 63] : '=');
    }
    return s;
}

bool sendAll(int fd, const uint8_t* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

int64_t nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

WsClient::WsClient() = default;
WsClient::~WsClient() { close(); }

void WsClient::close() {
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_RDWR);
        ::close(fd_);
        fd_ = -1;
    }
}

bool WsClient::waitReadable(int timeoutMs) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd_, &rfds);
    if (timeoutMs < 0) {
        return ::select(fd_ + 1, &rfds, nullptr, nullptr, nullptr) > 0;
    }
    struct timeval tv{};
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    return ::select(fd_ + 1, &rfds, nullptr, nullptr, &tv) > 0;
}

bool WsClient::readExact(uint8_t* out, size_t len, int timeoutMs) {
    const int64_t deadline = timeoutMs < 0 ? -1 : nowMs() + timeoutMs;
    size_t got = 0;
    while (got < len) {
        if (deadline >= 0) {
            int remain = (int)(deadline - nowMs());
            if (remain <= 0 || !waitReadable(remain)) {
                lastError_ = "read timeout";
                return false;
            }
        }
        ssize_t n = ::recv(fd_, out + got, len - got, 0);
        if (n <= 0) {
            lastError_ = "connection closed by peer";
            return false;
        }
        got += (size_t)n;
    }
    return true;
}

bool WsClient::connect(const std::string& host, uint16_t port,
                       const std::string& path, int timeoutMs) {
    close();
    sawClose_ = false;
    lastError_.clear();
    const int64_t deadline = nowMs() + timeoutMs;

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* list = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &list) != 0 || !list) {
        lastError_ = "dns resolve failed for '" + host + "'";
        return false;
    }
    int fd = -1;
    for (struct addrinfo* ai = list; ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int flags = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) {
            ::close(fd);
            fd = -1;
            continue;
        }
        int remain = (int)(deadline - nowMs());
        if (remain <= 0) {
            ::close(fd);
            fd = -1;
            break;
        }
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(fd, &wfds);
        struct timeval tv{};
        tv.tv_sec = remain / 1000;
        tv.tv_usec = (remain % 1000) * 1000;
        if (::select(fd + 1, nullptr, &wfds, nullptr, &tv) <= 0) {
            ::close(fd);
            fd = -1;
            continue;
        }
        int err = 0;
        socklen_t errLen = sizeof(err);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errLen) != 0 || err != 0) {
            ::close(fd);
            fd = -1;
            continue;
        }
        ::fcntl(fd, F_SETFL, flags); // back to blocking
        break;
    }
    ::freeaddrinfo(list);
    if (fd < 0) {
        lastError_ = "tcp connect failed to " + host + ":" + std::to_string(port);
        return false;
    }
    fd_ = fd;

    // --- HTTP upgrade handshake ---
    uint8_t keyBytes[16];
    std::random_device rd;
    for (int i = 0; i < 16; ++i) keyBytes[i] = static_cast<uint8_t>(rd());
    const std::string key = base64Encode(keyBytes, 16);
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Upgrade: websocket\r\n"
        << "Connection: Upgrade\r\n"
        << "Sec-WebSocket-Key: " << key << "\r\n"
        << "Sec-WebSocket-Version: 13\r\n\r\n";
    const std::string reqStr = req.str();
    if (!sendAll(fd_, reinterpret_cast<const uint8_t*>(reqStr.data()), reqStr.size())) {
        lastError_ = "handshake send failed";
        close();
        return false;
    }
    std::string resp;
    char buf[4096];
    const int64_t hsDeadline = deadline;
    while (resp.size() < 65536) {
        int remain = (int)(hsDeadline - nowMs());
        if (remain <= 0 || !waitReadable(remain)) {
            lastError_ = "handshake response timeout";
            close();
            return false;
        }
        ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0) {
            lastError_ = "handshake connection closed";
            close();
            return false;
        }
        resp.append(buf, (size_t)n);
        if (resp.find("\r\n\r\n") != std::string::npos) break;
    }
    const size_t eol = resp.find("\r\n");
    if (eol == std::string::npos || resp.find(" 101 ") == std::string::npos) {
        lastError_ = "handshake rejected: " + resp.substr(0, eol == std::string::npos ? 64 : eol);
        close();
        return false;
    }
    // Validate the accept key (RFC 6455 §4.1): proves the peer is a real
    // WebSocket server, not an HTTP server that echoed 101.
    static const char* kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const std::string keyIn = key + kGuid;
    Sha1 sha;
    sha.update(reinterpret_cast<const uint8_t*>(keyIn.data()), keyIn.size());
    uint8_t digest[20];
    sha.final(digest);
    const std::string expected = base64Encode(digest, 20);
    if (resp.find(expected) == std::string::npos) {
        lastError_ = "handshake accept-key mismatch";
        close();
        return false;
    }
    return true;
}

bool WsClient::sendBinary(const uint8_t* data, size_t len) {
    return sendBinary(std::vector<uint8_t>(data, data + len));
}

bool WsClient::sendBinary(const std::vector<uint8_t>& v) {
    if (fd_ < 0) {
        lastError_ = "not connected";
        return false;
    }
    // Client frames MUST be masked (RFC 6455 §5.3).
    uint8_t mask[4];
    std::random_device rd;
    for (int i = 0; i < 4; ++i) mask[i] = static_cast<uint8_t>(rd());
    uint8_t h[10];
    size_t hlen = 0;
    h[0] = 0x82; // FIN + binary
    const size_t n = v.size();
    if (n < 126) {
        h[1] = (uint8_t)(0x80 | n);
        hlen = 2;
    } else if (n < 65536) {
        h[1] = 0x80 | 126;
        h[2] = (uint8_t)((n >> 8) & 0xFF);
        h[3] = (uint8_t)(n & 0xFF);
        hlen = 4;
    } else {
        h[1] = 0x80 | 127;
        for (int i = 0; i < 8; ++i) h[2 + i] = (uint8_t)((n >> (56 - 8 * i)) & 0xFF);
        hlen = 10;
    }
    if (!sendAll(fd_, h, hlen)) {
        lastError_ = "send failed";
        return false;
    }
    if (!sendAll(fd_, mask, 4)) {
        lastError_ = "send failed";
        return false;
    }
    // Mask on the fly in stack-sized slices (meshes can be megabytes).
    uint8_t slice[8192];
    for (size_t off = 0; off < n; off += sizeof(slice)) {
        const size_t c = std::min(sizeof(slice), n - off);
        for (size_t i = 0; i < c; ++i)
            slice[i] = v[off + i] ^ mask[(off + i) % 4];
        if (!sendAll(fd_, slice, c)) {
            lastError_ = "send failed";
            return false;
        }
    }
    return true;
}

bool WsClient::recvMessage(std::vector<uint8_t>& out, int timeoutMs) {
    out.clear();
    if (fd_ < 0) {
        lastError_ = "not connected";
        return false;
    }
    std::vector<uint8_t> frag;
    for (;;) {
        uint8_t h[2];
        if (!readExact(h, 2, timeoutMs)) return false;
        const bool fin = (h[0] & 0x80) != 0;
        const uint8_t opcode = h[0] & 0x0F;
        const bool masked = (h[1] & 0x80) != 0;
        uint64_t len = h[1] & 0x7F;
        if (len == 126) {
            uint8_t e[2];
            if (!readExact(e, 2, timeoutMs)) return false;
            len = ((uint64_t)e[0] << 8) | e[1];
        } else if (len == 127) {
            uint8_t e[8];
            if (!readExact(e, 8, timeoutMs)) return false;
            len = 0;
            for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
        }
        if (len > (64u << 20)) {
            lastError_ = "frame too large";
            return false;
        }
        uint8_t mask[4] = {};
        if (masked && !readExact(mask, 4, timeoutMs)) return false;
        std::vector<uint8_t> payload((size_t)len);
        if (len && !readExact(payload.data(), (size_t)len, timeoutMs)) return false;
        if (masked) {
            for (uint64_t i = 0; i < len; ++i) payload[(size_t)i] ^= mask[i % 4];
        }
        if (opcode == 0x8) { // close
            sawClose_ = true;
            lastError_ = "server closed the connection";
            return false;
        }
        if (opcode == 0x9) { // ping -> pong (same payload)
            uint8_t ph[2] = {(uint8_t)(0x80 | 0xA), 0};
            size_t pl = (size_t)std::min<uint64_t>(len, 125);
            ph[1] = (uint8_t)(0x80 | pl);
            if (!sendAll(fd_, ph, 2)) return false;
            if (pl) {
                uint8_t pm[4] = {1, 2, 3, 4};
                if (!sendAll(fd_, pm, 4)) return false;
                for (size_t i = 0; i < pl; ++i) payload[i] ^= pm[i % 4];
                if (!sendAll(fd_, payload.data(), pl)) return false;
            }
            continue;
        }
        if (opcode == 0xA) continue; // pong
        if (opcode == 0x0) { // continuation
            frag.insert(frag.end(), payload.begin(), payload.end());
            if (!fin) continue;
            out.swap(frag);
            return true;
        }
        if (opcode == 0x1 || opcode == 0x2) {
            if (!fin) {
                frag = std::move(payload);
                continue;
            }
            out.swap(payload);
            return true;
        }
        lastError_ = "unexpected opcode";
        return false;
    }
}
