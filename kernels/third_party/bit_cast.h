/* Copyright 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See t-he License for the specific language governing permissions and
limitations under the License.
============================================================================= */
#ifndef CC_KERNEL_BIT_CAST_H
#define CC_KERNEL_BIT_CAST_H
#include <tops.h>

#include <type_traits>

#if __cplusplus >= 201402L
#define KERNEL_CONSTEXPR14 constexpr
#else
#define KERNEL_CONSTEXPR14
#endif

#if __cplusplus >= 201703L
#define KERNEL_CONSTEXPR17 constexpr
#define KERNEL_CONSTEXPR_IF if constexpr
#else
#define KERNEL_CONSTEXPR17
#define KERNEL_CONSTEXPR_IF if
#endif


// alternative to reinterpret_cast a value obeying strict aliasing rule
template <typename To, typename From>
__host__ __device__ __forceinline__ constexpr To bit_cast(const From& from) {
  static_assert(!std::is_pointer<typename std::decay<From>::type>::value &&
                    !std::is_pointer<typename std::decay<To>::type>::value &&
                    sizeof(From) == sizeof(To),
                "cannot cast through pointers or through different bits!");
  // typename std::remove_cv<To>::type to;
  // std::memcpy(&to, &from, sizeof(From));
  // return to;
  return __builtin_bit_cast(To, from);
}

// alternative to reinterpret_cast a pointer obeying strict aliasing rule
template <typename To, typename From>
__host__ __device__ __forceinline__ constexpr To pointer_cast(From&& from) {
  static_assert(
      std::is_pointer<typename std::decay<From>::type>::value &&
          std::is_pointer<To>::value &&
          sizeof(typename std::decay<From>::type) == sizeof(To),
      "pointer_cast supports only casting from one pointer type to another!!");

  static_assert(
      !std::is_const<typename std::remove_pointer<
              typename std::remove_reference<From>::type>::type>::value ||
          std::is_const<typename std::remove_pointer<To>::type>::value,
      "cannot cast from const pointer to non-const pointer!!");

  using TempPtr = typename std::conditional<
      std::is_const<typename std::remove_pointer<
          typename std::remove_reference<From>::type>::type>::value,
      const void*, void*>::type;

  return static_cast<To>(static_cast<TempPtr>(from));
}
#endif  // CC_KERNEL_BIT_CAST_H
