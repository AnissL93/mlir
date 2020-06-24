include_directories(src)
file(GLOB_RECURSE halide_src src/*.cpp)
add_library(halideir SHARED ${halide_src})
