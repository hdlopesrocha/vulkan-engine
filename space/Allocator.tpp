// Allocator.tpp - template implementations for Allocator<T>

#ifndef SPACE_ALLOCATOR_TPP
#define SPACE_ALLOCATOR_TPP

#include <cstdlib>
#include <iostream>

template <typename T>
void Allocator<T>::allocateBlock() {
    T* data = static_cast<T*>(std::malloc(blockSize * sizeof(T)));
    if (!data) throw std::bad_alloc();

    blocks.push_back({ data, totalAllocated });

    for (size_t i = 0; i < blockSize; ++i) {
        freeList.push_back(&data[i]);
        #ifndef NDEBUG
        deallocatedSet.insert(&data[i]);
        #endif
    }

    totalAllocated += blockSize;
}

template <typename T>
Allocator<T>::Allocator(size_t blockSize) : blockSize(blockSize) {
    std::cout << "Allocator(" << blockSize << ")" << std::endl;
}

template <typename T>
Allocator<T>::~Allocator() {
    for (auto &b : blocks) std::free(b.data);
}

template <typename T>
T* Allocator<T>::allocate() {
    std::unique_lock lock(mutex);
    if (freeList.empty()) allocateBlock();

    T* ptr = freeList.back();
    freeList.pop_back();

    #ifndef NDEBUG
    if (deallocatedSet.find(ptr) == deallocatedSet.end()) {
        throw std::runtime_error("Double allocate!");
    }
    deallocatedSet.erase(ptr);
    #endif
    return ptr;
}

template <typename T>
void Allocator<T>::deallocate(T* ptr) {
    if (!ptr) return;

    std::unique_lock lock(mutex);
    #ifndef NDEBUG
    if (deallocatedSet.find(ptr) != deallocatedSet.end()) {
        throw std::runtime_error("Double deallocate!");
    }
    #endif
    // check pointer belongs to a block
    bool valid = false;
    for (auto &b : blocks) {
        if (ptr >= b.data && ptr < b.data + blockSize) { valid = true; break; }
    }
    assert(valid && "Pointer does not belong to allocator");
    #ifndef NDEBUG
    deallocatedSet.insert(ptr);
    #endif
    freeList.push_back(ptr);
}

template <typename T>
uint Allocator<T>::getIndex(T* ptr) {
    if (!ptr) return UINT_MAX;

    std::shared_lock lock(mutex); // multiple allowed

    for (auto &b : blocks) {
        if (ptr >= b.data && ptr < b.data + blockSize) {
            return static_cast<uint>(b.startIndex + (ptr - b.data));
        }
    }
    throw std::runtime_error("Pointer does not belong to allocator");
}

template <typename T>
T* Allocator<T>::getFromIndex(uint index) {
    if (index == UINT_MAX) return nullptr;

    std::shared_lock lock(mutex); // multiple allowed
    uint blockIdx = index / blockSize;
    if (blockIdx >= blocks.size()) {
        throw std::runtime_error("Invalid index");
    }
    uint offset = index % blockSize;

    Block& b = blocks[blockIdx];

    T* ptr = b.data + offset;

    #ifndef NDEBUG
    if (deallocatedSet.find(ptr) != deallocatedSet.end()) {
        throw std::runtime_error("Accessing deallocated pointer");
    }
    #endif

    return ptr;
}

template <typename T>
void Allocator<T>::getFromIndices(T * nodes[8], uint indices[8]) {
    std::shared_lock lock(mutex); // multiple allowed        

    for(int i = 0 ; i < 8 ; ++i) {
        uint index = indices[i];
        if (index == UINT_MAX) {
            nodes[i] = NULL;
            continue;
        }

        uint blockIdx = index / blockSize;
        if (blockIdx >= blocks.size()) {
            throw std::runtime_error("Invalid index");
        }
        uint offset = index % blockSize;
        T* ptr = &blocks[blockIdx].data[offset];

        #ifndef NDEBUG
        if (deallocatedSet.find(ptr) != deallocatedSet.end()) {
            throw std::runtime_error("Accessing deallocated pointer");
        }
        #endif

        nodes[i] = ptr;
    }
}

template <typename T>
void Allocator<T>::reset() {
    std::unique_lock lock(mutex);
    freeList.clear();
    #ifndef NDEBUG
    deallocatedSet.clear();
    #endif
    for (auto &b : blocks) {
        for (size_t i = 0; i < blockSize; ++i) {
            T* ptr = &b.data[i];
            freeList.push_back(ptr);
            #ifndef NDEBUG
            deallocatedSet.insert(ptr);
            #endif
        }
    }
}

template <typename T>
uint Allocator<T>::allocateIndex() {
    T* ptr = allocate();
    return getIndex(ptr);
}

template <typename T>
size_t Allocator<T>::getAllocatedBlocksCount() {
    std::shared_lock lock(mutex);
    return blocks.size();
}

#endif // SPACE_ALLOCATOR_TPP
