#pragma once

// Per-frame GPU slot for the generic SDF renderer. Vulkan-coupled (VMA
// Buffer handles), so it lives here rather than in types/: only the plain
// data types (SdfStats, SdfProxyVertex, SdfEffectConfig, scene types) are
// in types/. Generic counterparts: types/SdfStats.hpp,
// types/SdfProxyVertex.hpp.

#include "../../resources/Buffer.hpp"
#include <cstdint>

// One slot per frame in flight: every buffer a draw reads is slot-local, so
// a rewrite for frame N can never race an in-flight draw of frame N-1/N-2
// that references a different slot (same pattern as
// DebugSDFRenderer::cullFrames). M9 (perf report 25): the scene vectors
// exist twice per slot. The plain `instance/.../gridIndex` buffers are the
// host-visible staging source (memcpy target, TRANSFER_SRC, never read by
// the march); the `gpu*` twins are the device-local march sources
// (TRANSFER_DST + STORAGE, bound to set=1 bindings 0..5). The single
// persistent host-visible buffer + host barrier alternative races when 3
// frames are in flight AND marches read over the host-visible heap; the
// staging -> device-local copy keeps triple-buffering AND device-local
// reads. `params` (binding 6) and `smoke` (binding 8) stay host-visible:
// tiny (<1 KiB together), streamed, HOST barrier only.
struct SdfFrameSlot {
    Buffer instance;    // staging: SdfInstance per container (memcpy source for binding 0)
    Buffer definition;  // staging: SdfDefinition (memcpy source for binding 1)
    Buffer material;    // staging: SdfMaterial (memcpy source for binding 2)
    Buffer container;   // staging: SdfContainer (memcpy source for binding 3)
    Buffer gridCell;    // staging: SdfGridCell (memcpy source for binding 4)
    Buffer gridIndex;   // staging: uint32_t (memcpy source for binding 5)
    // Device-local march sources (set=1 bindings 0..5). Descriptors point
    // here; rewritten only on (re)allocation, never per frame (VUID-03047).
    Buffer gpuInstance;   // device-local SdfInstance array (binding 0)
    Buffer gpuDefinition; // device-local SdfDefinition array (binding 1)
    Buffer gpuMaterial;   // device-local SdfMaterial array (binding 2)
    Buffer gpuContainer;  // device-local SdfContainer array (binding 3)
    Buffer gpuGridCell;   // device-local SdfGridCell array (binding 4)
    Buffer gpuGridIndex;  // device-local uint32_t array (binding 5)
    Buffer params;      // SdfParamsUBO uniform (binding 6)
    Buffer smoke;       // SmokeFragBullet: tuning + bullets + instance transform (binding 8)
    uint32_t instanceCap = 0;
    uint32_t definitionCap = 0;
    uint32_t materialCap = 0;
    uint32_t containerCap = 0;
    uint32_t gridCellCap = 0;
    uint32_t gridIndexCap = 0;
};
