#include "GltfWriter.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

// glTF requires JSON numbers where the schema says float (Blender's importer
// asserts on integers for accessor min/max), so always emit a decimal point.
std::string jsonFloat(float v) {
    if (!std::isfinite(v)) v = 0.0f;
    std::ostringstream ss;
    ss << v;
    std::string s = ss.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += ' ';
                } else {
                    out += c;
                }
                break;
        }
    }
    out += '"';
    return out;
}

struct Accessor {
    int bufferView = 0;
    int componentType = 5126;
    size_t count = 0;
    const char* type = "SCALAR";
    bool hasBounds = false;
    float min[4] = {0, 0, 0, 0};
    float max[4] = {0, 0, 0, 0};
};

constexpr int kComponentFloat = 5126;
constexpr int kComponentUint = 5125;
constexpr int kTargetArray = 34962;
constexpr int kTargetElement = 34963;

} // namespace

int GltfWriter::addImage(std::string name, std::string mimeType, std::vector<uint8_t> bytes) {
    Image img;
    img.name = std::move(name);
    img.mimeType = std::move(mimeType);
    img.bytes = std::move(bytes);
    images_.push_back(std::move(img));
    return static_cast<int>(images_.size()) - 1;
}

int GltfWriter::addMaterial(const Material& material) {
    materials_.push_back(material);
    return static_cast<int>(materials_.size()) - 1;
}

int GltfWriter::addMesh(std::string name, std::vector<Primitive> primitives) {
    Mesh mesh;
    mesh.name = std::move(name);
    mesh.primitives = std::move(primitives);
    meshes_.push_back(std::move(mesh));
    return static_cast<int>(meshes_.size()) - 1;
}

int GltfWriter::addNode(std::string name, int mesh) {
    Node node;
    node.name = std::move(name);
    node.mesh = mesh;
    nodes_.push_back(std::move(node));
    return static_cast<int>(nodes_.size()) - 1;
}

bool GltfWriter::write(const std::string& path, std::string& error) const {
    std::vector<uint8_t> bin;
    std::vector<Accessor> accessors;
    struct BufferView {
        size_t offset;
        size_t length;
        int target;
    };
    std::vector<BufferView> views;

    auto appendBytes = [&bin](const void* data, size_t bytes) -> size_t {
        const size_t offset = bin.size();
        const uint8_t* p = static_cast<const uint8_t*>(data);
        bin.insert(bin.end(), p, p + bytes);
        while ((bin.size() & 3u) != 0) bin.push_back(0);
        return offset;
    };

    // Adds one float accessor. `components` is 1..4, values are contiguous.
    auto addFloatAccessor = [&](const float* data, size_t count, int components,
                                const char* type, bool bounds, int target) -> int {
        const size_t bytes = count * static_cast<size_t>(components) * sizeof(float);
        const size_t offset = appendBytes(data, bytes);
        views.push_back({offset, bytes, target});
        Accessor acc;
        acc.bufferView = static_cast<int>(views.size()) - 1;
        acc.componentType = kComponentFloat;
        acc.count = count;
        acc.type = type;
        if (bounds && count > 0) {
            acc.hasBounds = true;
            for (int c = 0; c < components; ++c) {
                acc.min[c] = data[c];
                acc.max[c] = data[c];
            }
            for (size_t i = 1; i < count; ++i) {
                for (int c = 0; c < components; ++c) {
                    const float v = data[i * components + c];
                    if (v < acc.min[c]) acc.min[c] = v;
                    if (v > acc.max[c]) acc.max[c] = v;
                }
            }
        }
        accessors.push_back(acc);
        return static_cast<int>(accessors.size()) - 1;
    };

    auto addIndexAccessor = [&](const uint32_t* data, size_t count) -> int {
        const size_t bytes = count * sizeof(uint32_t);
        const size_t offset = appendBytes(data, bytes);
        views.push_back({offset, bytes, kTargetElement});
        Accessor acc;
        acc.bufferView = static_cast<int>(views.size()) - 1;
        acc.componentType = kComponentUint;
        acc.count = count;
        acc.type = "SCALAR";
        accessors.push_back(acc);
        return static_cast<int>(accessors.size()) - 1;
    };

    // Serialize meshes/primitives, collecting accessor indices.
    struct PrimitiveJson {
        std::string name;
        int mode = 4;
        int material = -1;
        std::vector<std::pair<std::string, int>> attributes;
        int indices = -1;
    };
    std::vector<std::vector<PrimitiveJson>> primitiveJsons(meshes_.size());
    for (size_t mi = 0; mi < meshes_.size(); ++mi) {
        const Mesh& mesh = meshes_[mi];
        for (const Primitive& prim : mesh.primitives) {
            PrimitiveJson pj;
            pj.name = prim.name;
            pj.mode = prim.mode;
            pj.material = prim.material;
            const size_t vcount = prim.positions.size() / 3;
            if (vcount == 0) continue;
            pj.attributes.push_back({"POSITION", addFloatAccessor(
                prim.positions.data(), vcount, 3, "VEC3", true, kTargetArray)});
            if (prim.normals.size() / 3 == vcount) {
                pj.attributes.push_back({"NORMAL", addFloatAccessor(
                    prim.normals.data(), vcount, 3, "VEC3", false, kTargetArray)});
            }
            if (prim.uvs.size() / 2 == vcount) {
                pj.attributes.push_back({"TEXCOORD_0", addFloatAccessor(
                    prim.uvs.data(), vcount, 2, "VEC2", false, kTargetArray)});
            }
            if (prim.octreeMat.size() / 4 == vcount) {
                pj.attributes.push_back({"_OCTREE_MAT", addFloatAccessor(
                    prim.octreeMat.data(), vcount, 4, "VEC4", false, kTargetArray)});
            }
            if (prim.veg.size() / 4 == vcount) {
                pj.attributes.push_back({"_VEG", addFloatAccessor(
                    prim.veg.data(), vcount, 4, "VEC4", false, kTargetArray)});
            }
            if (!prim.indices.empty()) {
                pj.indices = addIndexAccessor(prim.indices.data(), prim.indices.size());
            }
            primitiveJsons[mi].push_back(std::move(pj));
        }
    }

    // Embedded images.
    std::vector<int> imageViews;
    imageViews.reserve(images_.size());
    for (const Image& img : images_) {
        const size_t offset = appendBytes(img.bytes.data(), img.bytes.size());
        views.push_back({offset, img.bytes.size(), 0});
        imageViews.push_back(static_cast<int>(views.size()) - 1);
    }

    if (bin.empty()) bin.push_back(0);

    std::ostringstream json;
    json << "{";
    json << "\"asset\":{\"version\":\"2.0\",\"generator\":\"vulkan-engine Blender export\"}";
    json << ",\"scene\":0";
    json << ",\"scenes\":[{\"name\":\"Octree Export\",\"nodes\":[";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (i) json << ",";
        json << i;
    }
    json << "]}]";

    json << ",\"nodes\":[";
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node& n = nodes_[i];
        if (i) json << ",";
        json << "{\"name\":" << quote(n.name);
        if (n.mesh >= 0) json << ",\"mesh\":" << n.mesh;
        json << "}";
    }
    json << "]";

    json << ",\"meshes\":[";
    for (size_t mi = 0; mi < meshes_.size(); ++mi) {
        if (mi) json << ",";
        json << "{\"name\":" << quote(meshes_[mi].name) << ",\"primitives\":[";
        const std::vector<PrimitiveJson>& prims = primitiveJsons[mi];
        for (size_t pi = 0; pi < prims.size(); ++pi) {
            const PrimitiveJson& p = prims[pi];
            if (pi) json << ",";
            json << "{\"mode\":" << p.mode;
            if (p.material >= 0) json << ",\"material\":" << p.material;
            json << ",\"attributes\":{";
            for (size_t ai = 0; ai < p.attributes.size(); ++ai) {
                if (ai) json << ",";
                json << quote(p.attributes[ai].first) << ":" << p.attributes[ai].second;
            }
            json << "}";
            if (p.indices >= 0) json << ",\"indices\":" << p.indices;
            json << "}";
        }
        json << "]}";
    }
    json << "]";

    json << ",\"materials\":[";
    for (size_t i = 0; i < materials_.size(); ++i) {
        const Material& m = materials_[i];
        if (i) json << ",";
        json << "{\"name\":" << quote(m.name) << ",\"doubleSided\":"
             << (m.doubleSided ? "true" : "false");
        if (m.alphaBlend) json << ",\"alphaMode\":\"BLEND\"";
        json << ",\"pbrMetallicRoughness\":{";
        json << "\"baseColorFactor\":[" << jsonFloat(m.baseColor.r) << ","
             << jsonFloat(m.baseColor.g) << "," << jsonFloat(m.baseColor.b) << ","
             << jsonFloat(m.alphaBlend ? m.alpha : m.baseColor.a) << "]";
        json << ",\"metallicFactor\":" << jsonFloat(m.metallic);
        json << ",\"roughnessFactor\":" << jsonFloat(m.roughness);
        if (m.baseColorImage >= 0) {
            json << ",\"baseColorTexture\":{\"index\":" << m.baseColorImage << "}";
        }
        json << "}";
        if (m.normalImage >= 0) {
            json << ",\"normalTexture\":{\"index\":" << m.normalImage << ",\"scale\":1.0}";
        }
        const bool hasEmissive = m.emissive.r > 0.0f || m.emissive.g > 0.0f || m.emissive.b > 0.0f;
        if (hasEmissive) {
            json << ",\"emissiveFactor\":[" << jsonFloat(m.emissive.r) << ","
                 << jsonFloat(m.emissive.g) << "," << jsonFloat(m.emissive.b) << "]";
        }
        json << "}";
    }
    json << "]";

    // Textures are 1:1 with images (one shared sampler); the Blender script
    // resolves textures through these image names.
    json << ",\"textures\":[";
    for (size_t i = 0; i < images_.size(); ++i) {
        if (i) json << ",";
        json << "{\"sampler\":0,\"source\":" << i << "}";
    }
    json << "]";
    json << ",\"images\":[";
    for (size_t i = 0; i < images_.size(); ++i) {
        if (i) json << ",";
        json << "{\"name\":" << quote(images_[i].name) << ",\"mimeType\":"
             << quote(images_[i].mimeType) << ",\"bufferView\":" << imageViews[i] << "}";
    }
    json << "]";
    json << ",\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}]";

    json << ",\"accessors\":[";
    for (size_t i = 0; i < accessors.size(); ++i) {
        const Accessor& a = accessors[i];
        if (i) json << ",";
        json << "{\"bufferView\":" << a.bufferView
             << ",\"byteOffset\":0,\"componentType\":" << a.componentType
             << ",\"count\":" << a.count << ",\"type\":" << quote(a.type);
        if (a.hasBounds) {
            const int components = (std::strcmp(a.type, "VEC3") == 0) ? 3 :
                                   (std::strcmp(a.type, "VEC2") == 0) ? 2 : 1;
            json << ",\"min\":[";
            for (int c = 0; c < components; ++c) {
                if (c) json << ",";
                json << jsonFloat(a.min[c]);
            }
            json << "],\"max\":[";
            for (int c = 0; c < components; ++c) {
                if (c) json << ",";
                json << jsonFloat(a.max[c]);
            }
            json << "]";
        }
        json << "}";
    }
    json << "]";

    json << ",\"bufferViews\":[";
    for (size_t i = 0; i < views.size(); ++i) {
        const BufferView& v = views[i];
        if (i) json << ",";
        json << "{\"buffer\":0,\"byteOffset\":" << v.offset
             << ",\"byteLength\":" << v.length;
        if (v.target != 0) json << ",\"target\":" << v.target;
        json << "}";
    }
    json << "]";

    json << ",\"buffers\":[{\"byteLength\":" << bin.size() << "}]";
    json << "}";

    std::string jsonText = json.str();
    // JSON chunk padding is ASCII spaces; BIN chunk padding is zeros.
    while ((jsonText.size() & 3u) != 0) jsonText.push_back(' ');
    while ((bin.size() & 3u) != 0) bin.push_back(0);

    const uint32_t totalSize = static_cast<uint32_t>(
        12 + 8 + jsonText.size() + 8 + bin.size());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "cannot open '" + path + "' for writing";
        return false;
    }
    auto writeU32 = [&out](uint32_t v) {
        const uint8_t bytes[4] = {
            static_cast<uint8_t>(v & 0xFFu),
            static_cast<uint8_t>((v >> 8) & 0xFFu),
            static_cast<uint8_t>((v >> 16) & 0xFFu),
            static_cast<uint8_t>((v >> 24) & 0xFFu),
        };
        out.write(reinterpret_cast<const char*>(bytes), 4);
    };

    writeU32(0x46546C67u); // 'glTF'
    writeU32(2u);
    writeU32(totalSize);
    writeU32(static_cast<uint32_t>(jsonText.size()));
    writeU32(0x4E4F534Au); // 'JSON'
    out.write(jsonText.data(), static_cast<std::streamsize>(jsonText.size()));
    writeU32(static_cast<uint32_t>(bin.size()));
    writeU32(0x004E4942u); // 'BIN\0'
    out.write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(bin.size()));
    out.flush();
    if (!out.good()) {
        error = "write failed for '" + path + "'";
        return false;
    }
    return true;
}
