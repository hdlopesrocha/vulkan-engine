#pragma once

#include <glm/glm.hpp>

// The tessellated SURFACE POINT of one cell: a position and a normal, and
// nothing else. The material (brush index, HSV tint, texture-coordinate
// technique, UV scale) lives on the OctreeNode, so the walk never builds a
// Vertex. Runtime chunk vertices are computed on the GPU and never read back;
// only the offline Blender exporter supplies this point/normal from the cell's
// SDF.
struct CellVertex {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
};
