#pragma once

// Shared SHA-1 + Base64 for the WebSocket handshake (RFC 6455 §1.3/§4.1).
//
// One implementation used by BOTH ends so they cannot diverge:
// server/WebServer.cpp mints Sec-WebSocket-Accept with it, utils/WsClient.cpp
// validates with it. (A past copy-paste fork had a bitLen bug that made every
// real browser reject the handshake while raw-socket tests — which never
// check the key — passed. The unit test below pins the RFC test vector.)
//
// Header-only, dependency-free.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace wssha1 {

struct Sha1 {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint64_t bitLen = 0; // ORIGINAL message length in bits (see final())
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
            size_t take = len < 64 - bufLen ? len : 64 - bufLen;
            memcpy(buf + bufLen, data, take);
            bufLen += take; data += take; len -= take;
            if (bufLen == 64) { block(buf); bufLen = 0; }
        }
    }

    void final(uint8_t out[20]) {
        // The length field MUST encode the pre-padding message length, but
        // update() accumulates bitLen on every call — so snapshot it before
        // appending the 0x80 pad byte and zero padding. (Getting this wrong
        // yields a valid-looking but incorrect digest.)
        const uint64_t origBitLen = bitLen;
        uint8_t pad = 0x80, zero = 0x00;
        update(&pad, 1);
        while (bufLen != 56) update(&zero, 1);
        uint8_t lenBytes[8];
        for (int i = 0; i < 8; ++i) lenBytes[i] = (uint8_t)((origBitLen >> (56 - 8*i)) & 0xFF);
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

inline std::string base64Encode(const uint8_t* data, size_t len) {
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

// RFC 6455 §1.3: accept key for a client Sec-WebSocket-Key.
inline std::string wsAcceptKey(const std::string& clientKey) {
    static const char* kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const std::string in = clientKey + kGuid;
    Sha1 sha;
    sha.update(reinterpret_cast<const uint8_t*>(in.data()), in.size());
    uint8_t digest[20];
    sha.final(digest);
    return base64Encode(digest, 20);
}

} // namespace wssha1
