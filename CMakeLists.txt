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
#find_library(GLOG_LIB NAMES glog HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_library(NCCL_LIB nccl)

find_library(GFLAGS_LIB gflags)
#find_package(gflags REQUIRED)
#set (GFLAGS_LIB /usr/lib/x86_64-linux-gnu/libgflags.so.2)

find_package(CUDA REQUIRED)

find_package(Boost REQUIRED serialization)

find_package(Catch2 REQUIRED)

message(STATUS "Caffe2 lib: ${CAFFE2_LIB}")
message(STATUS "Caffe2 gpu lib: ${CAFFE2_GPU_LIB}")
message(STATUS "Caffe2 incl: ${CAFFE2_INCLUDE_DIRS}")
#message(STATUS "GLOG lib: ${GLOG_LIB} glog::glog")
message(STATUS "OpenCV lib: ${OpenCV_LIBRARIES}")
message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
message(STATUS "CUDA incl: ${CUDA_INCLUDE_DIRS}")
message(STATUS "cuRAND lib: ${CUDA_curand_LIBRARY}")
message(STATUS "Boost incl: ${Boost_INCLUDE_DIR}")
message(STATUS "GFlags: ${GFLAGS_LIB}")
message(STATUS "Proto: ${PROTOBUF_LIBRARY}")

# Main library for use with SLAM
add_library(hyper_op SHARED
  src/hyper_op.cc
)



foreach (lib hyper_op)
    target_include_directories(${lib}
      PUBLIC
      ${MY_CAFFE2_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIRS}
      ${Eigen_INCLUDE_DIRS}
      ${CAFFE2_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src/
      ${OpenCV_INCLUDE_DIRS})

  target_link_libraries(${lib}
    PUBLIC
      ${CAFFE2_LIB}
      ${CAFFE2_GPU_LIB}
      INTERFACE
    ${Boost_LIBRARIES}
      ${NCCL_LIB}
      ${CUDA_LIBRARIES}
      ${CUDA_curand_LIBRARY}
      ${CMAKE_THREAD_LIBS_INIT}
      ${GFLAGS_LIB}
      #${GLOG_LIB}
      ${PROTOBUF_LIBRARY}
      ${OpenCV_LIBRARIES})
endforeach(lib)


# A little test app
add_executable(tests
    src/hyper_op.cc
    src/test/test_main.cc
    src/test/test_op.cc
    #src/hyper_op.cc
)

target_link_libraries(tests
  #-Wl,--whole-archive hyper_op -Wl,--no-whole-archive
  hyper_op
  Catch2::Catch2
  )

target_include_directories(tests
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
