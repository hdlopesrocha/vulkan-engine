#include "PlaneModel.hpp"
#include <vector>

void PlaneModel::build(float width, float height, int hSegments, int vSegments, float texIndex) {
    int hSegs = std::max(1, hSegments);
    int vSegs = std::max(1, vSegments);

    float halfW = width * 0.5f;
    float halfH = height * 0.5f;

    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
int blendTestTex = 0;

    for (int y = 0; y <= vSegs; ++y) {
        for (int x = 0; x <= hSegs; ++x) {
            float u = (float)x / (float)hSegs;
            float v = (float)y / (float)vSegs;
            float vx = u * width - halfW;
            float vy = 0.0f;
            float vz = v * height - halfH;

            Vertex vert;
            vert.position = glm::vec3(vx, vy, vz);
            vert.color = glm::vec3(1.0f);
            vert.texCoord = glm::vec2(u, v);
            vert.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vert.texIndex = static_cast<int>(blendTestTex++ % 2);
            vert.tangent = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
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
