#pragma once

// Per-frame GPU slot for the generic SDF renderer. Vulkan-coupled (VMA
// Buffer handles), so it lives here rather than in types/: only the plain
// data types (SdfStats, SdfProxyVertex, SdfEffectConfig, scene types) are
// in types/. Generic counterparts: types/SdfStats.hpp,
// types/SdfProxyVertex.hpp.

#include "../Buffer.hpp"
#include <cstdint>

// One slot per frame in flight: every buffer a draw reads is slot-local,
// so a host rewrite for frame N can never race an in-flight draw of frame
// N-1/N-2 that references a different slot. Static scene data could share
// one buffer, but per-slot copies keep the lifetime story trivial (same
// pattern as DebugSDFRenderer::cullFrames) at negligible memory cost.
// The single persistent buffer + host barrier alternative would race when
// 3 frames are in flight (frame N+1 rewrite vs frame N draw), which is why
// triple-buffering is used instead.
struct SdfFrameSlot {
    Buffer instance;    // SdfGpuInstance per container (set=1 binding 0)
    Buffer definition;  // SdfGpuDefinition (binding 1)
    Buffer material;    // SdfGpuMaterial (binding 2)
    Buffer container;   // SdfGpuContainer (binding 3)
    Buffer gridCell;    // SdfGpuGridCell (binding 4)
    Buffer gridIndex;   // uint32_t (binding 5)
    Buffer params;      // SdfParamsUBO uniform (binding 6)
    Buffer smoke;       // SmokeFragBulletGPU: tuning + bullets (binding 8)
    uint32_t instanceCap = 0;
    uint32_t definitionCap = 0;
    uint32_t materialCap = 0;
    uint32_t containerCap = 0;
    uint32_t gridCellCap = 0;
    uint32_t gridIndexCap = 0;
};
