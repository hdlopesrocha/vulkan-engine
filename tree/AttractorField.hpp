#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <random>

namespace tree {

struct Attractor {
    glm::vec3 position;
    bool alive = true;
};

class AttractorField {
public:
    std::vector<Attractor> points;

    void clear() { points.clear(); }

    void populateSphere(const glm::vec3& center, float radius, int count, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        points.reserve(points.size() + count);
        for (int i = 0; i < count; ++i) {
            // Rejection sampling inside sphere
            glm::vec3 p;
            do {
                p = glm::vec3(dist(rng), dist(rng), dist(rng));
            } while (glm::dot(p, p) > 1.0f);
            points.push_back({center + p * radius, true});
        }
    }

    void populateBox(const glm::vec3& min, const glm::vec3& max, int count, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        glm::vec3 size = max - min;
        points.reserve(points.size() + count);
        for (int i = 0; i < count; ++i) {
            glm::vec3 p = min + size * glm::vec3(dist(rng), dist(rng), dist(rng));
            points.push_back({p, true});
        }
    }

    // Tree crown: oblate (flattened) hemisphere — wider than tall, only above baseY.
    // radiusXZ controls lateral spread, radiusY controls crown thickness.
    void populateCrown(const glm::vec3& center, float radiusXZ, float radiusY,
                       int count, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        points.reserve(points.size() + count);
        for (int i = 0; i < count; ++i) {
            glm::vec3 p;
            do {
                p = glm::vec3(dist(rng), dist(rng), dist(rng));
            } while (glm::dot(p, p) > 1.0f || p.y < 0.0f);
            points.push_back({center + p * glm::vec3(radiusXZ, radiusY, radiusXZ), true});
        }
    }

    // Full ellipsoid (no hemisphere constraint) — for layered branching at all heights.
    void populateEllipsoid(const glm::vec3& center, float radiusXZ, float radiusY,
                           int count, unsigned seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        points.reserve(points.size() + count);
        for (int i = 0; i < count; ++i) {
            glm::vec3 p;
            do {
                p = glm::vec3(dist(rng), dist(rng), dist(rng));
            } while (glm::dot(p, p) > 1.0f);
            points.push_back({center + p * glm::vec3(radiusXZ, radiusY, radiusXZ), true});
        }
    }

    size_t aliveCount() const {
        size_t n = 0;
        for (auto& a : points) if (a.alive) ++n;
        return n;
    }

    size_t size() const { return points.size(); }
};

} // namespace tree
