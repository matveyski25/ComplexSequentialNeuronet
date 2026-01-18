//
// Created by matve on 05.01.2026.
//

#ifndef CSN_TEMPLATELIMITS_H
#define CSN_TEMPLATELIMITS_H

#pragma once
#include <type_traits>

namespace MyNN::Utils::TemplateLimits {
    template<typename Derived, typename Base, typename Enable = std::enable_if_t<std::is_base_of_v<Derived, Base> > >
    class ExtendIs {
    };

    template<typename Derived, typename... Bases>
    class ExtendsIs : ExtendIs<Derived, Bases>... {
    };

    template<typename T, typename Enable = std::enable_if_t<std::is_arithmetic_v<T> > >
    class IsArithmeticType {
    };

    template<typename... Ts>
    class IsArithmeticTypes : IsArithmeticType<Ts>... {
    };
}

namespace MyNN::Utils::MyPtr {
    using std::unique_ptr;

    template <class _Ty, class _Dx = std::default_delete<_Ty>>
    class copy_ptr {
        std::unique_ptr<_Ty, _Dx> ptr;

    public:
        copy_ptr() = default;
        explicit copy_ptr(std::unique_ptr<_Ty> p) : ptr(std::move(p)) {}

        copy_ptr(const copy_ptr& other)
            : ptr(other.ptr ? other.ptr->clone() : nullptr) {}

        //explicit operator std::unique_ptr<T>() {
        //    return std::move(ptr);
        //}

        copy_ptr& operator=(const copy_ptr& other) {
            if (this != &other)
                ptr = other.ptr ? other.ptr->clone() : nullptr;
            return *this;
        }

        copy_ptr(copy_ptr&&) noexcept = default;
        copy_ptr& operator=(copy_ptr&&) noexcept = default;

        _Ty* get() const { return ptr.get(); }
        _Ty& operator*() const { return *ptr; }
        _Ty* operator->() const { return ptr.get(); }
    };
}

#endif //CSN_TEMPLATELIMITS_H