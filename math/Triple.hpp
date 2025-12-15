// Auto-generated wrapper header for Triple
#pragma once

#include "Common.hpp"

template<typename A, typename B, typename C>
struct Triple {
    A first;
    B second;
    C third;
    bool operator==(const Triple& other) const {
        return first == other.first && second == other.second && third == other.third;
    }
};