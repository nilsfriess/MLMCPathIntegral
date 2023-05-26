cmake_minimum_required(VERSION 3.12)

project(MLMCPathIntegral VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_library(MLMCPathIntegral INTERFACE)
target_include_directories(MLMCPathIntegral INTERFACE include)
target_include_directories(MLMCPathIntegral SYSTEM INTERFACE external/blaze)
target_compile_features(MLMCPathIntegral INTERFACE cxx_std_20)
target_compile_options(MLMCPathIntegral INTERFACE -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused -O3 -march=native) 

find_package(LAPACK REQUIRED)
target_link_libraries(MLMCPathIntegral INTERFACE LAPACK::LAPACK)

add_executable(harmonic_oscillator examples/harmonic_oscillator.cc)
target_include_directories(harmonic_oscillator SYSTEM PRIVATE external/json/single_include)
target_link_libraries(harmonic_oscillator PRIVATE MLMCPathIntegral)

add_executable(harmonic_oscillator_two_level examples/harmonic_oscillator_two_level.cc)
target_include_directories(harmonic_oscillator_two_level PRIVATE external/json/single_include)
target_link_libraries(harmonic_oscillator_two_level PRIVATE MLMCPathIntegral)

# add_executable(test_partition tests/partition_test.cc)
# target_link_libraries(test_partition PRIVATE MLMCPathIntegral)
 
