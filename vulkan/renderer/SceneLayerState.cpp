#include "SceneLayerState.hpp"

#include <algorithm>
#include <thread>

SceneLayerState::SceneLayerState()
    : mutex(std::make_unique<std::recursive_mutex>()),
      genPool(std::make_unique<ThreadPool>(
          std::max(2u, std::thread::hardware_concurrency() / 2))) {}
