add_executable(banana banana.cc)
target_link_libraries(banana PRIVATE LAPACK::LAPACK blaze::blaze MLMCPathIntegral)

add_executable(harmonic_oscillator harmonic_oscillator.cc)
find_package(TBB REQUIRED)
target_link_libraries(harmonic_oscillator PRIVATE LAPACK::LAPACK blaze::blaze TBB::tbb MLMCPathIntegral)
