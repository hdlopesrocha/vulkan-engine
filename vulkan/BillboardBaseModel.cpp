#include "BillboardBaseModel.hpp"

BillboardBaseModel::BillboardBaseModel() {
    createBase();
}

BillboardBaseModel::~BillboardBaseModel() {}

const std::vector<Vertex>& BillboardBaseModel::getVertices() const {
    return vertices;
}

const std::vector<uint32_t>& BillboardBaseModel::getIndices() const {
    return indices;
}

void BillboardBaseModel::createBase() {
    // Simple quad in XZ plane, centered at origin, Y is up
    // Size: 2x2 units (can be changed as needed)
    vertices = std::vector<Vertex>{
        Vertex({-1.0f, 0.0f, -1.0f}, {0,1,0}, {0,0}, 0),
        Vertex({ 1.0f, 0.0f, -1.0f}, {0,1,0}, {1,0}, 0),
        Vertex({ 1.0f, 0.0f,  1.0f}, {0,1,0}, {1,1}, 0),
        Vertex({-1.0f, 0.0f,  1.0f}, {0,1,0}, {0,1}, 0),
    };
    indices = std::vector<uint32_t>{
        0, 1, 2,
        2, 3, 0
    };
}
