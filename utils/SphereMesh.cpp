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
            vert.pos[0] = px; vert.pos[1] = py; vert.pos[2] = pz;
            vert.color[0] = 1.0f; vert.color[1] = 1.0f; vert.color[2] = 1.0f;
            vert.uv[0] = u; vert.uv[1] = v;
            vert.normal[0] = px / radius; vert.normal[1] = py / radius; vert.normal[2] = pz / radius;
            vert.texIndex = texIndex;
            vert.tangent[0] = -sinTheta; vert.tangent[1] = 0.0f; vert.tangent[2] = cosTheta; vert.tangent[3] = 1.0f;
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
