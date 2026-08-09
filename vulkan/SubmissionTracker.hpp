#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

// Ring of recent queue submissions, shared by all submit sites. When a fence
// wait exceeds a few seconds (amdgpu watchdog territory), dump() shows exactly
// which tagged jobs were submitted last, so a GPU ring hang is attributable to
// the job at the head of the ring.
namespace SubmissionTracker {

struct Entry {
    uint64_t seq;
    uint64_t wallMs;      // steady-clock ms since process start (relative order)
    uint64_t wallClockS;  // system-clock epoch seconds (correlates with dmesg/journalctl)
    char tag[24];
};

inline constexpr uint32_t kEntries = 128;

inline std::atomic<uint32_t>& index() { static std::atomic<uint32_t> i{0}; return i; }
inline Entry* entries() { static Entry e[kEntries] = {}; return e; }
inline std::atomic<uint64_t>& seqCounter() { static std::atomic<uint64_t> s{0}; return s; }

inline uint64_t record(const char* tag) {
    uint32_t i = index().fetch_add(1) % kEntries;
    Entry& e = entries()[i];
    e.seq = seqCounter().fetch_add(1) + 1;
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    e.wallMs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
    auto wall = std::chrono::system_clock::now().time_since_epoch();
    e.wallClockS = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(wall).count());
    std::snprintf(e.tag, sizeof(e.tag), "%s", tag);
    return e.seq;
}

inline void dump() {
    // std::cerr is unbuffered: a stall dump must survive process death (the
    // amdgpu watchdog kill can happen mid-frame, dropping buffered stdout).
    std::cerr << "[submission] recent queue submissions (oldest first, wall=SysClock):\n";
    uint32_t i = index().load() % kEntries;
    for (uint32_t n = 0; n < kEntries; ++n) {
        Entry& e = entries()[(i + n) % kEntries];
        if (e.seq == 0) continue;
        std::cerr << "[submission]   seq=" << e.seq << " t=" << e.wallMs << "ms wall=" << e.wallClockS << "s tag=" << e.tag << "\n";
    }
    std::cerr.flush();
}

} // namespace SubmissionTracker
