cmake_minimum_required(VERSION 3.10)
project(tinyexpr VERSION 1.0)

# save all libs to the same directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)

# build for tinyexpr
if(WIN32)
    add_library(${PROJECT_NAME} STATIC "tinyexpr.c")
else()
    add_library(${PROJECT_NAME} SHARED "tinyexpr.c")
endif()
if(NOT WIN32)
    target_link_libraries(${PROJECT_NAME} m)
endif()

# build samples for tinyexpr
if(NOT DEFINED VP_BUILD_FROM)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/samples)  # save all exe to 'samples'
    add_executable(tinyexpr_test "tinyexpr_test.cpp")
    target_link_libraries(tinyexpr_test ${PROJECT_NAME})
endif()
