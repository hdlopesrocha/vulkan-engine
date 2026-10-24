#pragma once

#include "OctreeGpuNode.hpp"
#include "OctreeTopologyInput.hpp"

// There is no host-side triangle stream: the pass writes Cartesian vertices
// only into the renderer's device-local packed pools. The host side sees just
// the submitted node/SDF tables and the integer counts.
