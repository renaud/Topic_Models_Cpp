cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8.0)

project(TopicModel)

find_package(MiniBuola REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(SYSTEM
            ${EIGEN3_INCLUDE_DIR}
)

#FILE(GLOB_RECURSE cc_sources *.cc)
set(cc_sources ccorpus.cc cmodel.cc main.cc opt.cc)

add_executable(TopicModel ${cc_sources})
add_definitions(-std=c++11 -DNDEBUG)

target_link_libraries(TopicModel buolamini gsl gslcblas nlopt)

install(TARGETS TopicModel DESTINATION /usr/local/bin)
