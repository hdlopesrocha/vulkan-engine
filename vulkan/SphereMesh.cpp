#include "SphereMesh.hpp"
#include <vector>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// Build a UV sphere using the specified parameters.
void SphereMesh::build(VulkanApp* app, float radius, int slices, int stacks, float texIndex) {
    // Ensure sensible defaults
    if (slices < 3) slices = 3;
    if (stacks < 2) stacks = 2;

    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    // Precompute rings of vertices (stacks + 1 rows, slices + 1 columns)
    for (int y = 0; y <= stacks; ++y) {
        float v = float(y) / float(stacks);
        float theta = v * glm::pi<float>(); // [0, pi]
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int x = 0; x <= slices; ++x) {
            float u = float(x) / float(slices);
            float phi = u * glm::two_pi<float>(); // [0, 2pi]
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            glm::vec3 pos = glm::vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi) * radius;
            glm::vec3 n = glm::normalize(pos);
            glm::vec2 uv = glm::vec2(u, 1.0f - v); // flip v for conventional texture orientation

            Vertex vert{};
            vert.pos[0] = pos.x; vert.pos[1] = pos.y; vert.pos[2] = pos.z;
            vert.color[0] = 1.0f; vert.color[1] = 1.0f; vert.color[2] = 1.0f;
            vert.uv[0] = uv.x; vert.uv[1] = uv.y;
            vert.normal[0] = n.x; vert.normal[1] = n.y; vert.normal[2] = n.z;
            vert.texIndex = texIndex;
            vert.tangent[0] = 0.0f; vert.tangent[1] = 0.0f; vert.tangent[2] = 0.0f; vert.tangent[3] = 1.0f;
            vertices.push_back(vert);
        }
    }

    // Build indices (two triangles per quad)
    for (int y = 0; y < stacks; ++y) {
        for (int x = 0; x < slices; ++x) {
            uint16_t i0 = uint16_t(y * (slices + 1) + x);
            uint16_t i1 = uint16_t((y + 1) * (slices + 1) + x);
            uint16_t i2 = uint16_t((y + 1) * (slices + 1) + (x + 1));
            uint16_t i3 = uint16_t(y * (slices + 1) + (x + 1));

            // first triangle
            indices.push_back(i0);
            indices.push_back(i1);
            indices.push_back(i2);
            // second triangle
            indices.push_back(i2);
            indices.push_back(i3);
            indices.push_back(i0);
        }
    }

    // Build GPU buffers; compute per-vertex tangents on CPU from positions/UVs
    Model3D::build(app, vertices, indices, false);
}
