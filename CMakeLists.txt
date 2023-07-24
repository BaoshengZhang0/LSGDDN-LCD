cmake_minimum_required(VERSION 2.8)
project(LSGDDN)
SET(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  ")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

# find lib
find_package(OpenCV 3.2.0 QUIET)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
   ${PROJECT_SOURCE_DIR}/include
   ${PROJECT_SOURCE_DIR}/lib/yaml-cpp
   ${OpenCV_INCLUDE_DIRS}
   ${EIGEN3_INCLUDE_DIR}
)
set(SRC_LISTS
        src/main.cpp
        src/LSGD.cpp
        src/DynamicNode.cpp
        include/tools.cpp)

set(LINK_LIBS
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        yaml-cpp)

add_executable(LSGDDN ${SRC_LISTS} )
target_link_libraries(LSGDDN ${LINK_LIBS})

