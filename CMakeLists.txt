cmake_minimum_required(VERSION 3.4)

project(HyperOp)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# Convenience
list(APPEND Eigen_INCLUDE_DIRS /usr/include/eigen3)

list(APPEND OpenCV_LIB opencv_xfeatures2d)

#add_definitions(-fPIC -lpthread -g)
add_definitions(-lpthread -g)

# Caffe2
# If built from pytorch git
find_library(CAFFE2_LIB caffe2 REQUIRED)
find_library(CAFFE2_GPU_LIB caffe2_gpu REQUIRED)
find_library(GLOG_LIB NAMES glog HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(GFLAGS_LIB gflags)
find_library(NCCL_LIB nccl)

find_package(CUDA REQUIRED)

find_package(Boost REQUIRED serialization)

message(STATUS "Caffe2 lib: ${CAFFE2_LIB}")
message(STATUS "Caffe2 gpu lib: ${CAFFE2_GPU_LIB}")
message(STATUS "Caffe2 incl: ${CAFFE2_INCLUDE_DIRS}")
message(STATUS "GLOG lib: ${GLOG_LIB} glog::glog")
message(STATUS "OpenCV lib: ${OpenCV_LIBRARIES}")
message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
message(STATUS "CUDA incl: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuRAND lib: ${CUDA_curand_LIBRARY}")
message(STATUS "Boost incl: ${Boost_INCLUDE_DIR}")

# Main library for use with SLAM
add_library(hyper_op SHARED
  src/hyper_op.cc
)



# There's PUBLIC INTERFACE and PRIVATE
target_include_directories(hyper_op
    PUBLIC
    ${MY_CAFFE2_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${CAFFE2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(hyper_op
  ${Boost_LIBRARIES}
    ${GFLAGS_LIB}
    ${NCCL_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${PROTOBUF_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLOG_LIB}
    ${CAFFE2_LIB}
    ${CAFFE2_GPU_LIB}
    ${OpenCV_LIBRARIES})

# A little test app
add_executable(test_op
    src/test/test_op.cc
    src/hyper_op.cc
)

target_link_libraries(test_op
  ${Boost_LIBRARIES}
    ${GFLAGS_LIB}
    ${NCCL_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${PROTOBUF_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    ${GLOG_LIB}
    ${CAFFE2_LIB}
    ${CAFFE2_GPU_LIB}
    ${OpenCV_LIBRARIES}
    hyper_op)

target_include_directories(test_op
    PUBLIC
    ${MY_CAFFE2_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${CAFFE2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
    ${OpenCV_INCLUDE_DIRS})


# CMake is the worst monstrosity I've ever seen.
# Copy includes for external library use.
#[[
file(GLOB_RECURSE LIBRARY_INCLUDES ${PROJECT_SOURCE_DIR}/include/**.h)
message(STATUS ${PROJECT_SOURCE_DIR}/include)
message(STATUS ${LIBRARY_INCLUDES})
add_custom_command(TARGET app POST_BUILD
  COMMAND /usr/bin/cmake -E copy ${LIBRARY_INCLUDES} ${PROJECT_SOURCE_DIR}/build/cpp_slam/
)
#]]
