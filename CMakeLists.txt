cmake_minimum_required(VERSION 3.12)

project(MLMCPathIntegral VERSION 0.0.1 LANGUAGES CXX)

add_library(MLMCPathIntegral INTERFACE)
target_include_directories(MLMCPathIntegral INTERFACE include)
target_include_directories(MLMCPathIntegral SYSTEM INTERFACE external/blaze)
target_compile_features(MLMCPathIntegral INTERFACE cxx_std_20)
target_compile_options(MLMCPathIntegral INTERFACE -Wall -Wextra -Wpedantic)

find_package(LAPACK REQUIRED)
target_link_libraries(MLMCPathIntegral INTERFACE LAPACK::LAPACK)

add_executable(harmonic_oscillator apps/harmonic_oscillator.cc)
target_include_directories(harmonic_oscillator PRIVATE external/json/single_include)
target_link_libraries(harmonic_oscillator PRIVATE MLMCPathIntegral)                    
 
