#include "OctreeRaymarchTable.hpp"

#include <bit>
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

// Store `child` in `slot` of `parent`'s compact child header.
//
// The builder appends a parent's Surface children consecutively in slot order,
// so a node only needs the first child index (meta.y) and an 8-bit presence
// mask (hsv1.w). The assert below is the contiguity contract: if a child is
// not the next index after the already-stored ones, the layout is corrupt and
// the shader's popcount resolution would address the wrong node.
void OctreeRaymarchTable::setChild(uint32_t parent, int slot, int32_t child) {
    assert(parent < nodes.size());
    assert(slot >= 0 && slot < 8);
    // slot already validated; reading the mask as an exact integer is safe
    // because the builder only ever stores integers 0..255 in hsv1.w.
    uint32_t mask = static_cast<uint32_t>(nodes[parent].hsv1.w + 0.5f);
    if (mask == 0) {
        nodes[parent].meta.y = child;
    } else {
        assert(child == nodes[parent].meta.y + std::popcount(mask));
    }
    mask |= 1u << slot;
    nodes[parent].hsv1.w = static_cast<float>(mask);
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
