#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// Minimal glTF 2.0 binary (.glb) writer, self-contained (no external
// dependencies beyond glm). Supports exactly what the Blender exporter needs:
//   - one buffer with 4-byte aligned bufferViews,
//   - triangle and point primitives,
//   - float and uint32 accessors,
//   - custom FLOAT vertex attributes for material indices and tints
//     (Blender 4.x imports float custom attributes; integer custom attributes
//     abort the importer, so all application attributes are floats whose
//     values are exactly integral for indices),
//   - embedded images (JPEG/PNG passed through untouched),
//   - PBR materials (baseColor + normal textures) for generic glTF viewers;
//     the Blender script builds the real triplanar blend materials.
class GltfWriter {
public:
    struct Material {
        std::string name;
        glm::vec4 baseColor{1.0f};
        float metallic = 0.0f;
        float roughness = 0.85f;
        glm::vec3 emissive{0.0f};
        float emissiveStrength = 1.0f;
        int baseColorImage = -1; // addImage() index, -1 = none
        int normalImage = -1;
        bool doubleSided = true;
        bool alphaBlend = false;
        float alpha = 1.0f;
    };

    struct Primitive {
        std::string name;
        int mode = 4; // 4 = TRIANGLES, 0 = POINTS
        int material = -1;
        // Interleaved-by-attribute float arrays (xyz, xyz, uv, ...).
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;
        // Application attributes. Blender 4.0's glTF importer aborts on
        // custom attributes with *different component counts* in the same
        // primitive, so each mesh carries at most one packed vec4:
        //   _OCTREE_MAT = (texture index, hsv.x, hsv.y, hsv.z)   [terrain]
        //   _VEG        = (variant, rotation, height scale, 0)   [grass]
        std::vector<float> octreeMat;    // 4 floats per vertex
        std::vector<float> veg;          // 4 floats per vertex
        std::vector<uint32_t> indices;   // empty for POINTS
    };

    int addImage(std::string name, std::string mimeType, std::vector<uint8_t> bytes);
    int addMaterial(const Material& material);
    // Returns the node index.
    int addMesh(std::string name, std::vector<Primitive> primitives);
    // Adds a scene root node referencing `mesh` (-1 = empty node).
    int addNode(std::string name, int mesh);

    // Writes the GLB. Returns false and fills `error` on I/O or validation
    // failure.
    bool write(const std::string& path, std::string& error) const;

    size_t vertexCount() const;
    size_t triangleCount() const;

private:
    struct Image {
        std::string name;
        std::string mimeType;
        std::vector<uint8_t> bytes;
    };
    struct Mesh {
        std::string name;
        std::vector<Primitive> primitives;
    };
    struct Node {
        std::string name;
        int mesh = -1;
    };

    std::vector<Image> images_;
    std::vector<Material> materials_;
    std::vector<Mesh> meshes_;
    std::vector<Node> nodes_;
};
