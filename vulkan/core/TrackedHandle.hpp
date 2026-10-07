#pragma once

#include <vulkan/vulkan.h>

template<typename T>
struct TrackedHandle {
    T handle = VK_NULL_HANDLE;

    ~TrackedHandle() { handle = VK_NULL_HANDLE; }

    TrackedHandle() = default;
    TrackedHandle(T h) : handle(h) {}

    TrackedHandle& operator=(T h) { handle = h; return *this; }

    operator T() const { return handle; }

    bool operator==(T h) const { return handle == h; }
    bool operator!=(T h) const { return handle != h; }

    T* operator&() { return &handle; }
    const T* operator&() const { return &handle; }
};