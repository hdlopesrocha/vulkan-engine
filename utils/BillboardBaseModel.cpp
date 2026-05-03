#include "BillboardBaseModel.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

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

void BillboardBaseModel::addQuad(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3) {
    uint32_t baseIdx = static_cast<uint32_t>(vertices.size());
    
    glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    
    vertices.push_back(Vertex(v0, normal, {0.0f, 1.0f}, 0));
    vertices.push_back(Vertex(v1, normal, {1.0f, 1.0f}, 0));
    vertices.push_back(Vertex(v2, normal, {1.0f, 0.0f}, 0));
    vertices.push_back(Vertex(v3, normal, {0.0f, 0.0f}, 0));
    
    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 1);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 3);
    indices.push_back(baseIdx + 0);
}

void BillboardBaseModel::createBase() {
    // Create 6 planes for 3D billboard:
    // - 4 VERTICAL planes tilted 45° from center (rotated 0°, 90°, 180°, 270° around Y)
    //   Each plane leans outward so when viewing from ANY angle, it's never parallel to view vector
    // - 2 vertical planes perpendicular to ground (vertical cross pattern at top)
    
    const float halfWidth = 0.5f;   // Width of each plane at base (±0.5 in X or Z)
    const float height = 1.0f;      // Height from ground (0 to 1 in Y)
    const float tiltDistance = 0.707f;  // 45° tilt = 0.707 * height forward/backward
    
    // --- 4 TILTED VERTICAL PLANES (45° from center, rotated 0/90/180/270°) ---
    
    // Plane 0: Faces forward-north (Z negative), leans outward
    // Bottom plane: Z = -0.5, Top plane: Z = -1.207 (tilted 45° back)
    addQuad(
        glm::vec3(-halfWidth, 0.0f, -0.5f),                    // bottom-left
        glm::vec3( halfWidth, 0.0f, -0.5f),                    // bottom-right
        glm::vec3( halfWidth, height, -0.5f - tiltDistance),   // top-right (tilted back 45°)
        glm::vec3(-halfWidth, height, -0.5f - tiltDistance)    // top-left (tilted back 45°)
    );
    
    // Plane 1: Faces right-east (X positive), leans outward
    // Bottom plane: X = 0.5, Top plane: X = 1.207 (tilted 45° outward)
    addQuad(
        glm::vec3( 0.5f, 0.0f, -halfWidth),                    // bottom-left
        glm::vec3( 0.5f, 0.0f,  halfWidth),                    // bottom-right
        glm::vec3( 0.5f + tiltDistance, height,  halfWidth),   // top-right (tilted right 45°)
        glm::vec3( 0.5f + tiltDistance, height, -halfWidth)    // top-left (tilted right 45°)
    );
    
    // Plane 2: Faces backward-south (Z positive), leans outward
    // Bottom plane: Z = 0.5, Top plane: Z = 1.207 (tilted 45° forward)
    addQuad(
        glm::vec3( halfWidth, 0.0f,  0.5f),                    // bottom-left
        glm::vec3(-halfWidth, 0.0f,  0.5f),                    // bottom-right
        glm::vec3(-halfWidth, height,  0.5f + tiltDistance),   // top-right (tilted forward 45°)
        glm::vec3( halfWidth, height,  0.5f + tiltDistance)    // top-left (tilted forward 45°)
    );
    
    // Plane 3: Faces left-west (X negative), leans outward
    // Bottom plane: X = -0.5, Top plane: X = -1.207 (tilted 45° outward)
    addQuad(
        glm::vec3(-0.5f, 0.0f,  halfWidth),                    // bottom-left
        glm::vec3(-0.5f, 0.0f, -halfWidth),                    // bottom-right
        glm::vec3(-0.5f - tiltDistance, height, -halfWidth),   // top-right (tilted left 45°)
        glm::vec3(-0.5f - tiltDistance, height,  halfWidth)    // top-left (tilted left 45°)
    );
    
    // --- 2 VERTICAL PLANES (cross pattern, perpendicular to each other) ---
    
    // Plane 4: Vertical, aligned with X-axis (facing Z direction)
    // Creates vertical plane in ZY plane
    addQuad(
        glm::vec3(-halfWidth, 0.0f, -0.5f),     // bottom-left
        glm::vec3( halfWidth, 0.0f, -0.5f),     // bottom-right
        glm::vec3( halfWidth, height, -0.5f),   // top-right
        glm::vec3(-halfWidth, height, -0.5f)    // top-left
    );
    
    // Plane 5: Vertical, aligned with Z-axis (facing X direction)
    // Creates vertical plane in XY plane
    addQuad(
        glm::vec3(-0.5f, 0.0f, -halfWidth),     // bottom-left
        glm::vec3(-0.5f, 0.0f,  halfWidth),     // bottom-right
        glm::vec3(-0.5f, height,  halfWidth),   // top-right
        glm::vec3(-0.5f, height, -halfWidth)    // top-left
    );
}

