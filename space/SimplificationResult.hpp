#pragma once

#include <cstdint>

// Result of attempting to simplify eight child nodes into their parent.
// `isSimplified` indicates whether simplification succeeded.
struct SimplificationResult {
    uint8_t isSimplified;
    SimplificationResult(uint8_t simplified)
        : isSimplified(simplified) {
    }
};
