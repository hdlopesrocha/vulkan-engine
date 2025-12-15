#ifndef SPACE_ALLOCATOR_HPP
#define SPACE_ALLOCATOR_HPP

#include <vector>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <cstdlib>
#include <climits>
#include <stdexcept>
#include <iostream>
#define NDEBUG 1

template <typename T>
class Allocator {
private:
    struct Block {
        T* data;          // ponteiro para os elementos
        size_t startIndex; // índice global do primeiro elemento deste bloco
    };

    std::vector<Block> blocks;
    std::vector<T*> freeList;
    #ifndef NDEBUG
    std::unordered_set<T*> deallocatedSet;
    #endif
    const size_t blockSize;
    size_t totalAllocated = 0;
    mutable std::shared_mutex mutex;    

    void allocateBlock();

public:
    Allocator(size_t blockSize);

    ~Allocator();

    // -------------------
    // Alocação / Liberação
    // -------------------
    T* allocate();

    void deallocate(T* ptr);

    // -------------------
    // Index <-> Pointer O(1)
    // -------------------
    uint getIndex(T* ptr);

    T* getFromIndex(uint index);


    void getFromIndices(T * nodes[8], uint indices[8]);

    void reset();

    uint allocateIndex();

    size_t getAllocatedBlocksCount();

    size_t getBlockSize() const { return blockSize; }
};

#include "Allocator.tpp"
#endif // SPACE_ALLOCATOR_HPP

