add_executable(banana banana.cc)
target_link_libraries(banana PRIVATE LAPACK::LAPACK blaze::blaze MLMCPathIntegral)
