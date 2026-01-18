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

    template<typename Context>
    struct IFeature {
        using Context_ = Context;
    };
}


#endif //CSN_TEMPLATELIMITS_H