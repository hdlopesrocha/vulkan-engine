#ifndef SPACE_HPP
#define SPACE_HPP

// Thin umbrella header for space module. Heavy declarations are split into focused headers.

#include <semaphore>
#include <thread>
#include <future>
#include <tsl/robin_map.h>
#include <unordered_set>
#include <utility>
#include <shared_mutex>
#include "../math/math.hpp"
#include "../sdf/SDF.hpp"
#define SQRT_3_OVER_2 0.866025404f
#include "ThreadPool.hpp"
#include "Allocator.hpp"
#include "OctreeAllocator.hpp"

#include "types.hpp"
#include "node.hpp"
#include "octree.hpp"
#include "iterator.hpp"

#endif // SPACE_HPP
