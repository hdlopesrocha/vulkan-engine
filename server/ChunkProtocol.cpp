#include "ChunkProtocol.hpp"
#include <zlib.h>
#include <cstring>
#include <stdexcept>

namespace chunkproto {
namespace {

inline void putU16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
}
inline void putU32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 16) & 0xFF));
    v.push_back(static_cast<uint8_t>((x >> 24) & 0xFF));
}
inline void putU64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>((x >> (8*i)) & 0xFF));
}
inline void putF32(std::vector<uint8_t>& v, float f) {
    uint32_t u;
    static_assert(sizeof(u) == sizeof(f));
    std::memcpy(&u, &f, sizeof(f));
    putU32(v, u);
}
inline void putI32(std::vector<uint8_t>& v, int32_t i) {
    putU32(v, static_cast<uint32_t>(i));
}

} // namespace

std::vector<uint8_t> gzipCompress(const uint8_t* data, size_t len) {
    z_stream strm{};
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("gzipCompress: deflateInit2 failed");
    std::vector<uint8_t> out;
    out.reserve(len / 2 + 64);
    uint8_t buf[16384];
    strm.next_in = const_cast<Bytef*>(data);
    strm.avail_in = static_cast<uInt>(len);
    int flush = Z_FINISH;
    int ret;
    do {
        strm.next_out = buf;
        strm.avail_out = sizeof(buf);
        ret = deflate(&strm, flush);
        if (ret == Z_STREAM_ERROR) {
            deflateEnd(&strm);
            throw std::runtime_error("gzipCompress: deflate failed");
        }
        out.insert(out.end(), buf, buf + (sizeof(buf) - strm.avail_out));
    } while (ret != Z_STREAM_END);
    deflateEnd(&strm);
    return out;
}

std::vector<uint8_t> gzipDecompress(const uint8_t* data, size_t len) {
    if (len == 0) return {};
    z_stream strm{};
    // 47 = auto-detect gzip/zlib (16+MAX_WBITS is gzip-only; 47 handles both).
    if (inflateInit2(&strm, 47) != Z_OK)
        throw std::runtime_error("gzipDecompress: inflateInit2 failed");
    std::vector<uint8_t> out;
    out.reserve(len * 3 + 64);
    uint8_t buf[16384];
    strm.next_in = const_cast<Bytef*>(data);
    strm.avail_in = static_cast<uInt>(len);
    int ret;
    do {
        strm.next_out = buf;
        strm.avail_out = sizeof(buf);
        ret = inflate(&strm, Z_NO_FLUSH);
        if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
            inflateEnd(&strm);
            throw std::runtime_error("gzipDecompress: inflate failed");
        }
        out.insert(out.end(), buf, buf + (sizeof(buf) - strm.avail_out));
    } while (ret != Z_STREAM_END);
    inflateEnd(&strm);
    return out;
}

std::vector<uint8_t> buildRequestAll() {
    return std::vector<uint8_t>{MSG_REQUEST_ALL};
}

std::vector<uint8_t> buildRequestMesh(uint64_t chunkId) {
    std::vector<uint8_t> v;
    v.reserve(9);
    v.push_back(MSG_REQUEST_MESH);
    for (int i = 0; i < 8; ++i)
        v.push_back(static_cast<uint8_t>((chunkId >> (8 * i)) & 0xFF));
    return v;
}

std::vector<uint8_t> buildSnapshotStart(uint32_t total) {
    std::vector<uint8_t> v;
    v.reserve(9);
    v.push_back('C'); v.push_back('H'); v.push_back('K'); v.push_back('1');
    v.push_back(MSG_SNAPSHOT_START);
    putU32(v, total);
    return v;
}

std::vector<uint8_t> buildSnapshotEnd() {
    return std::vector<uint8_t>{'C','H','K','1', MSG_SNAPSHOT_END};
}

std::vector<uint8_t> buildChunkBatch(const ChunkRecord* recs, size_t count) {
    std::vector<uint8_t> v;
    v.reserve(7 + count * CHUNK_RECORD_SIZE);
    v.push_back('C'); v.push_back('H'); v.push_back('K'); v.push_back('1');
    v.push_back(MSG_CHUNK_BATCH);
    putU16(v, static_cast<uint16_t>(count));
    uint8_t tmp[CHUNK_RECORD_SIZE];
    for (size_t i = 0; i < count; ++i) {
        encodeRecord(recs[i], tmp);
        v.insert(v.end(), tmp, tmp + CHUNK_RECORD_SIZE);
    }
    return v;
}

std::vector<uint8_t> buildDeleteBatch(const uint64_t* ids, size_t count) {
    std::vector<uint8_t> v;
    v.reserve(7 + count * 8);
    v.push_back('C'); v.push_back('H'); v.push_back('K'); v.push_back('1');
    v.push_back(MSG_DELETE_BATCH);
    putU16(v, static_cast<uint16_t>(count));
    for (size_t i = 0; i < count; ++i) putU64(v, ids[i]);
    return v;
}

std::vector<uint8_t> buildSceneMeta(const SceneMetaLayer* layers, size_t count) {
    std::vector<uint8_t> v;
    v.reserve(9 + count * 24);
    v.push_back('C'); v.push_back('H'); v.push_back('K'); v.push_back('1');
    v.push_back(MSG_SCENE_META);
    putU32(v, static_cast<uint32_t>(count));
    for (size_t i = 0; i < count; ++i) {
        const SceneMetaLayer& m = layers[i];
        v.push_back(m.layer);
        putF32(v, m.rootMinX); putF32(v, m.rootMinY); putF32(v, m.rootMinZ);
        putF32(v, m.rootLen);
        putF32(v, m.chunkSize);
        v.push_back(m.rootChunkLod); v.push_back(0); v.push_back(0);
    }
    return v;
}

std::vector<uint8_t> buildMeshData(const MeshData& mesh) {
    const uint32_t nVerts = static_cast<uint32_t>(mesh.positions.size() / 3);
    const uint32_t nIdx = static_cast<uint32_t>(mesh.indices.size());
    std::vector<uint8_t> v;
    v.reserve(24 + mesh.positions.size() * 4 + mesh.indices.size() * 4);
    v.push_back('C'); v.push_back('H'); v.push_back('K'); v.push_back('1');
    v.push_back(MSG_MESH_DATA);
    putU64(v, mesh.id);
    putU32(v, mesh.version);
    v.push_back(mesh.layer); v.push_back(0); v.push_back(0); v.push_back(0);
    putU32(v, nVerts);
    putU32(v, nIdx);
    for (float f : mesh.positions) putF32(v, f);
    for (uint32_t i : mesh.indices) putU32(v, i);
    return v;
}

void encodeRecord(const ChunkRecord& r, uint8_t* o) {
    // Little-endian explicit writes; floats via bit copy (IEEE754 LE on x86).
    auto wU16 = [&](size_t off, uint16_t x) {
        o[off] = static_cast<uint8_t>(x & 0xFF);
        o[off+1] = static_cast<uint8_t>((x >> 8) & 0xFF);
    };
    auto wU32 = [&](size_t off, uint32_t x) {
        o[off] = static_cast<uint8_t>(x & 0xFF);
        o[off+1] = static_cast<uint8_t>((x >> 8) & 0xFF);
        o[off+2] = static_cast<uint8_t>((x >> 16) & 0xFF);
        o[off+3] = static_cast<uint8_t>((x >> 24) & 0xFF);
    };
    auto wU64 = [&](size_t off, uint64_t x) {
        for (int i = 0; i < 8; ++i) o[off+i] = static_cast<uint8_t>((x >> (8*i)) & 0xFF);
    };
    auto wF32 = [&](size_t off, float f) {
        uint32_t u; std::memcpy(&u, &f, 4); wU32(off, u);
    };
    wU64(0, r.id);
    wF32(8, r.minX); wF32(12, r.minY); wF32(16, r.minZ);
    wF32(20, r.size);
    wU32(24, r.level);
    o[28] = r.lod; o[29] = r.chunkLod; o[30] = r.isLeaf; o[31] = r.spaceType;
    o[32] = r.layer; o[33] = 0; o[34] = 0; o[35] = 0;
    wU32(36, r.version);
    wU32(40, static_cast<uint32_t>(r.brushIndex));
    for (int i = 0; i < 8; ++i) wF32(44 + 4*i, r.sdf[i]);
    (void)wU16;
}

bool decodeRecord(const uint8_t* in, size_t avail, ChunkRecord& out) {
    if (avail < CHUNK_RECORD_SIZE) return false;
    auto rU32 = [&](size_t off) -> uint32_t {
        return (uint32_t)in[off] | ((uint32_t)in[off+1] << 8) |
               ((uint32_t)in[off+2] << 16) | ((uint32_t)in[off+3] << 24);
    };
    auto rU64 = [&](size_t off) -> uint64_t {
        uint64_t x = 0;
        for (int i = 0; i < 8; ++i) x |= (uint64_t)in[off+i] << (8*i);
        return x;
    };
    auto rF32 = [&](size_t off) -> float {
        uint32_t u = rU32(off); float f; std::memcpy(&f, &u, 4); return f;
    };
    out.id = rU64(0);
    out.minX = rF32(8); out.minY = rF32(12); out.minZ = rF32(16);
    out.size = rF32(20);
    out.level = rU32(24);
    out.lod = in[28]; out.chunkLod = in[29]; out.isLeaf = in[30]; out.spaceType = in[31];
    out.layer = in[32];
    out.version = rU32(36);
    out.brushIndex = static_cast<int32_t>(rU32(40));
    for (int i = 0; i < 8; ++i) out.sdf[i] = rF32(44 + 4*i);
    return true;
}

} // namespace chunkproto
