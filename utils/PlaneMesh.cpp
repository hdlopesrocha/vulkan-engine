#include "PlaneMesh.hpp"
#include <vector>

void PlaneMesh::build(float width, float height, int hSegments, int vSegments, float texIndex) {
    int hSegs = std::max(1, hSegments);
    int vSegs = std::max(1, vSegments);

    float halfW = width * 0.5f;
    float halfH = height * 0.5f;

    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    for (int y = 0; y <= vSegs; ++y) {
        for (int x = 0; x <= hSegs; ++x) {
            float u = (float)x / (float)hSegs;
            float v = (float)y / (float)vSegs;
            float vx = u * width - halfW;
            float vy = 0.0f;
            float vz = v * height - halfH;

            Vertex vert;
            vert.pos[0] = vx; vert.pos[1] = vy; vert.pos[2] = vz;
            vert.color[0] = 1.0f; vert.color[1] = 1.0f; vert.color[2] = 1.0f;
            vert.uv[0] = u; vert.uv[1] = v;
            vert.normal[0] = 0.0f; vert.normal[1] = 1.0f; vert.normal[2] = 0.0f;
            vert.texIndex = texIndex;
            vert.tangent[0] = 0.0f; vert.tangent[1] = 0.0f; vert.tangent[2] = 1.0f; vert.tangent[3] = 1.0f;
            vertices.push_back(vert);
        }
    }

    for (int y = 0; y < vSegs; ++y) {
        for (int x = 0; x < hSegs; ++x) {
            int a = y * (hSegs + 1) + x;
            int b = a + hSegs + 1;
            indices.push_back(a);
            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b);
            indices.push_back(a + 1);
            indices.push_back(b + 1);
        }
    }

    setGeometry(vertices, indices);
    computeTangents();
}
