add_executable(example example.cc)

find_package(LAPACK)
find_package(blaze REQUIRED)

target_include_directories(example PRIVATE "../external/gcem/install/include")
target_include_directories(example PRIVATE "../external/stats/include")

target_link_libraries(example PRIVATE LAPACK::LAPACK blaze::blaze MLMCPathIntegral)
