#include "BoxLineGeometry.hpp"
#include "BoundingBox.hpp"
#include "Math.hpp"

BoxLineGeometry::BoxLineGeometry(const BoundingBox &box) : Geometry() {
    glm::vec3 corners[8] = {
        {box.getMinX(), box.getMinY(), box.getMinZ()},
        {box.getMaxX(), box.getMinY(), box.getMinZ()},
        {box.getMaxX(), box.getMaxY(), box.getMinZ()},
        {box.getMinX(), box.getMaxY(), box.getMinZ()},
        {box.getMinX(), box.getMinY(), box.getMaxZ()},
        {box.getMaxX(), box.getMinY(), box.getMaxZ()},
        {box.getMaxX(), box.getMaxY(), box.getMaxZ()},
        {box.getMinX(), box.getMaxY(), box.getMaxZ()},
    };
    
    // Texture coordinates for each corner (0=min, 1=max in each dimension)
    // This creates a proper mapping for grid texture on cube edges
    glm::vec3 texCoords[8] = {
        {0.0f, 0.0f, 0.0f},  // min corner
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},  // max corner
        {0.0f, 1.0f, 1.0f},
    };
    
    // Add 8 unique vertices with explicit texture coordinates
    for(int i = 0; i < 8; ++i) {
        Vertex v(corners[i], glm::normalize(corners[i] - box.getCenter()), texCoords[i], 0);
        addVertex(v);
    }
    
    // Add indices for 12 lines (24 indices for line pairs)
    uint lineIndices[24] = {
        0, 1, 1, 2, 2, 3, 3, 0, // bottom face
        4, 5, 5, 6, 6, 7, 7, 4, // top face
        0, 4, 1, 5, 2, 6, 3, 7  // vertical edges
    };
    
    for(int i = 0; i < 24; ++i) {
        indices.push_back(lineIndices[i]);
    }
}

