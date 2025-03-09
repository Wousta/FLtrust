cmake_minimum_required(VERSION 3.22.1)
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

set(Torch_DIR "/home/bustaman/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(lyra REQUIRED)

add_library(mnistTrain STATIC src/mnistTrain.cpp)
add_executable(srvr src/srvr.cpp)
add_executable(clnt src/clnt.cpp)

target_compile_options(srvr PRIVATE -g)
target_compile_options(clnt PRIVATE -g)

target_include_directories(mnistTrain PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include 
  ${TORCH_INCLUDE_DIRS}
)
target_include_directories(srvr PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(clnt PUBLIC ${lyra_INCLUDE_DIRS})
target_include_directories(srvr PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(mnistTrain PUBLIC ${TORCH_LIBRARIES})

target_link_libraries(srvr PUBLIC  
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${TORCH_LIBRARIES}
  mnistTrain  
)

target_link_libraries(clnt PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/../RcConn/build/librc_conn.so
  ${CMAKE_CURRENT_SOURCE_DIR}/../rdma-api/build/libnorm.so
  ${TORCH_LIBRARIES}
  mnistTrain
)