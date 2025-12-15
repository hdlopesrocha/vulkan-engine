#pragma once

#include <cstdint>
#include "OctreeNodeData.hpp"

struct StackFrame : public OctreeNodeData {
    uint8_t childIndex;
    uint8_t internalOrder[8];
    bool secondVisit;

    StackFrame(const OctreeNodeData &data, uint8_t childIndex, bool secondVisit) : OctreeNodeData(data) {
        this->childIndex = childIndex;
        this->secondVisit = secondVisit;
    }
};