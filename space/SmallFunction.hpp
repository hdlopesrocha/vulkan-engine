#pragma once

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>
#include <functional>

/// A move-only type-erased callable wrapper with small-buffer optimization.
/// Stores callables up to 48 bytes inline; larger callables fall back to heap.
class SmallFunction {
    static constexpr size_t InlineSize = 48;

    struct VTable {
        void (*invoke)(void* self);
        void (*destroy)(void* self);
        void (*move)(void* src, void* dst);
    };

    alignas(alignof(std::max_align_t)) char storage_[InlineSize];
    const VTable* vtable_ = nullptr;

public:
    SmallFunction() = default;

    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, SmallFunction>)
    SmallFunction(F&& f)
    {
        using T = std::decay_t<F>;
        static constexpr bool canInline = sizeof(T) <= InlineSize
            && alignof(T) <= alignof(std::max_align_t)
            && std::is_nothrow_move_constructible_v<T>;

        static constexpr VTable vt = {
            [](void* self) {
                if constexpr (canInline) {
                    std::invoke(*static_cast<T*>(self));
                } else {
                    std::invoke(**static_cast<T**>(self));
                }
            },
            [](void* self) {
                if constexpr (canInline) {
                    static_cast<T*>(self)->~T();
                } else {
                    delete* static_cast<T**>(self);
                }
            },
            [](void* src, void* dst) {
                if constexpr (canInline) {
                    ::new (dst) T(std::move(*static_cast<T*>(src)));
                    static_cast<T*>(src)->~T();
                } else {
                    *static_cast<T**>(dst) = *static_cast<T**>(src);
                }
            }
        };

        if constexpr (canInline) {
            ::new (storage_) T(std::forward<F>(f));
        } else {
            *reinterpret_cast<T**>(storage_) = new T(std::forward<F>(f));
        }
        vtable_ = &vt;
    }

    SmallFunction(SmallFunction&& other) noexcept
    {
        if (other.vtable_) {
            other.vtable_->move(other.storage_, storage_);
            vtable_ = other.vtable_;
            other.vtable_ = nullptr;
        }
    }

    SmallFunction& operator=(SmallFunction&& other) noexcept
    {
        if (this != &other) {
            if (vtable_) vtable_->destroy(storage_);
            vtable_ = other.vtable_;
            if (other.vtable_) {
                other.vtable_->move(other.storage_, storage_);
                other.vtable_ = nullptr;
            }
        }
        return *this;
    }

    SmallFunction(const SmallFunction&) = delete;
    SmallFunction& operator=(const SmallFunction&) = delete;

    ~SmallFunction()
    {
        if (vtable_) vtable_->destroy(storage_);
    }

    void operator()()
    {
        if (vtable_) vtable_->invoke(storage_);
    }

    explicit operator bool() const noexcept
    {
        return vtable_ != nullptr;
    }
};
