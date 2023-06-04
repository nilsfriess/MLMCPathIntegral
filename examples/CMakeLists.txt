# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
   cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(harmonic_oscillator harmonic_oscillator.cc)
target_link_libraries(harmonic_oscillator PRIVATE MLMCPathIntegral nlohmann_json::nlohmann_json)

add_executable(harmonic_oscillator_two_level harmonic_oscillator_two_level.cc)
target_link_libraries(harmonic_oscillator_two_level PRIVATE MLMCPathIntegral nlohmann_json::nlohmann_json)

add_executable(harmonic_oscillator_multilevel harmonic_oscillator_multilevel.cc)
target_link_libraries(harmonic_oscillator_multilevel PRIVATE MLMCPathIntegral nlohmann_json::nlohmann_json)

add_executable(banana banana.cc)
target_link_libraries(banana PRIVATE LAPACK::LAPACK)
 
