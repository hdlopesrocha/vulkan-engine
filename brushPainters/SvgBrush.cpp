#include "SvgBrush.hpp"

#include <cmath>

SvgBrush::SvgBrush(int materialIndex, glm::vec3 fallbackHsv)
    : m_material(materialIndex)
    , m_fallbackHsv(fallbackHsv) {}

int SvgBrush::paint(const OctreeNode &o) const {
    (void)o;
    return m_material;
}

glm::vec3 SvgBrush::paintHSV(const OctreeNode &o) const {
    // The per-position fill tint needs the surface position, which the octree
    // build no longer computes (no SDF::getPosition), so the fallback tint is
    // used. Re-add the position on the node / move the tint to the GPU to
    // restore per-contour colours.
    (void)o;
    return m_fallbackHsv;
}
