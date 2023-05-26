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

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
   cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(harmonic_oscillator examples/harmonic_oscillator.cc)
target_link_libraries(harmonic_oscillator PRIVATE MLMCPathIntegral nlohmann_json::nlohmann_json)

add_executable(harmonic_oscillator_two_level examples/harmonic_oscillator_two_level.cc)
target_link_libraries(harmonic_oscillator_two_level PRIVATE MLMCPathIntegral nlohmann_json::nlohmann_json)

add_executable(banana examples/banana.cc)
target_link_libraries(banana PRIVATE LAPACK::LAPACK)
 
