#include "OctreeRaymarchTable.hpp"

#include <cassert>

void OctreeRaymarchTable::clear() {
    nodes.clear();
    layerRoots.clear();
}

uint32_t OctreeRaymarchTable::append(const OctreeRaymarchNode& node) {
    const uint32_t index = static_cast<uint32_t>(nodes.size());
    nodes.push_back(node);
    return index;
}

void OctreeRaymarchTable::setChild(uint32_t parent, int slot, int32_t child) {
    assert(parent < nodes.size());
    assert(slot >= 0 && slot < 8);
    glm::ivec4& packed = (slot < 4) ? nodes[parent].child03 : nodes[parent].child47;
    packed[slot & 3] = child;
}

void OctreeRaymarchTable::setPrunedChild(uint32_t parent, int slot) {
    assert(parent < nodes.size());
    assert(slot >= 0 && slot < 8);
    nodes[parent].meta.w |= static_cast<int32_t>(1u << (PRUNED_CHILD_SHIFT + slot));
}

void OctreeRaymarchTable::setChildType(uint32_t parent, int slot, uint32_t type) {
    assert(parent < nodes.size());
    assert(slot >= 0 && slot < 8);
    nodes[parent].meta.w |= static_cast<int32_t>((type & 3u) << (CHILD_TYPE_SHIFT + slot * 2));
}
