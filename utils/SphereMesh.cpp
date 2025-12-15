#include "SphereMesh.hpp"
#include <vector>
#include <cmath>

void SphereMesh::build(float radius, int longitudes, int latitudes, float texIndex) {
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;

    for (int y = 0; y <= latitudes; ++y) {
        for (int x = 0; x <= longitudes; ++x) {
            float u = (float)x / (float)longitudes;
            float v = (float)y / (float)latitudes;
            float theta = u * 2.0f * M_PI;
            float phi = v * M_PI;

            float sinPhi = sin(phi);
            float cosPhi = cos(phi);
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);

            float px = radius * sinPhi * cosTheta;
            float py = radius * cosPhi;
            float pz = radius * sinPhi * sinTheta;

            Vertex vert;
            vert.position = glm::vec3(px, py, pz);
            vert.color = glm::vec3(1.0f);
            vert.texCoord = glm::vec2(u, v);
            vert.normal = glm::vec3(px / radius, py / radius, pz / radius);
            vert.texIndex = static_cast<int>(texIndex);
            vert.tangent = glm::vec4(-sinTheta, 0.0f, cosTheta, 1.0f);
            vertices.push_back(vert);
        }
    }

    for (int y = 0; y < latitudes; ++y) {
        for (int x = 0; x < longitudes; ++x) {
            int a = y * (longitudes + 1) + x;
            int b = a + longitudes + 1;
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
            indices.push_back(a + 1);
        }
    }

    setGeometry(vertices, indices);
    computeTangents();
}
