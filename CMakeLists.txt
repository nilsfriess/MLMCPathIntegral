cmake_minimum_required(VERSION 3.12)

project(MLMCPathIntegral VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_library(MLMCPathIntegral INTERFACE)
target_include_directories(MLMCPathIntegral INTERFACE include)

target_compile_features(MLMCPathIntegral INTERFACE cxx_std_20)
target_compile_options(MLMCPathIntegral INTERFACE -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused -O3 -march=native) 

# Maybe we will use a different linear algebra lib in the future
add_compile_definitions(USE_BLAZE)

target_include_directories(MLMCPathIntegral SYSTEM INTERFACE external/blaze)
find_package(LAPACK REQUIRED)
target_link_libraries(MLMCPathIntegral INTERFACE LAPACK::LAPACK)

find_program(CCACHE_PATH ccache)
if (CCACHE_PATH)
    message(STATUS "CCache found in ${CCACHE_PATH}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif (CCACHE_PATH)

add_subdirectory(examples)
 
