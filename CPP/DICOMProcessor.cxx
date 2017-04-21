cmake_minimum_required(VERSION 2.8)

project(DICOMProcessor)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(DICOMProcessor DICOMProcessor.cxx)

target_link_libraries(DICOMProcessor ${ITK_LIBRARIES})