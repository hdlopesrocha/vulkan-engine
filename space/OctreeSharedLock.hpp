#pragma once
#include <shared_mutex>

// Read guard for Octree::treeMutex. iterate* can nest (an iterate handler may
// re-enter the octree, e.g. LocalScene::requestModel3D calls the triangle
// walker from inside a traversal handler), and std::shared_mutex is not
// recursive, so only the OUTERMOST iterate acquires the lock; inner ones (same
// thread) piggyback on the outer read. apply takes the exclusive path (it is
// the only writer) and is never re-entered from an iterate handler.
//
// Lives in a header so every translation unit that re-enters the octree
// (Octree.cpp itself and the CPU/headless triangle walker) shares one
// thread_local depth counter; a per-TU copy would double lock_shared and
// deadlock on the first nested call.
class OctreeSharedLock {
public:
    explicit OctreeSharedLock(std::shared_mutex &m) : m_(m) {
        if (depth_++ == 0) m_.lock_shared();
    }
    ~OctreeSharedLock() {
        if (--depth_ == 0) m_.unlock_shared();
    }
    OctreeSharedLock(const OctreeSharedLock&) = delete;
    OctreeSharedLock& operator=(const OctreeSharedLock&) = delete;
private:
    static inline thread_local int depth_ = 0;
    std::shared_mutex &m_;
};
