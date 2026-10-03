#include "Frustum.hpp"
#include "Math.hpp"

Frustum::Frustum(glm::mat4 m)
{
	m = glm::transpose(m);
	m_planes[Left]   = m[3] + m[0];
	m_planes[Right]  = m[3] - m[0];
	m_planes[Bottom] = m[3] + m[1];
	m_planes[Top]    = m[3] - m[1];
	m_planes[Near]   = m[3] + m[2];
	m_planes[Far]    = m[3] - m[2];

}

ContainmentType Frustum::test(const AbstractBoundingBox &box) {
    glm::vec3 minp = box.getMin();
    glm::vec3 maxp = box.getMax();

    // Build the 8 corner points of the AABB (homogeneous w=1)
    glm::vec4 corners[8] = {
        glm::vec4(minp.x, minp.y, minp.z, 1.0f),
        glm::vec4(maxp.x, minp.y, minp.z, 1.0f),
        glm::vec4(minp.x, maxp.y, minp.z, 1.0f),
        glm::vec4(maxp.x, maxp.y, minp.z, 1.0f),
        glm::vec4(minp.x, minp.y, maxp.z, 1.0f),
        glm::vec4(maxp.x, minp.y, maxp.z, 1.0f),
        glm::vec4(minp.x, maxp.y, maxp.z, 1.0f),
        glm::vec4(maxp.x, maxp.y, maxp.z, 1.0f)
    };

    bool allInside = true;

    // For each frustum plane, count how many AABB corners are outside (dot < 0)
    for (int i = 0; i < Count; ++i) {
        int outside = 0;
        for (int j = 0; j < 8; ++j) {
            if (glm::dot(m_planes[i], corners[j]) < 0.0f) ++outside;
        }

        // If all corners are outside of a plane, the box is disjoint
        if (outside == 8) return ContainmentType::Disjoint;

        if (outside > 0) allInside = false;
    }

    if (allInside) return ContainmentType::Contains;

    return ContainmentType::Intersects;
}


