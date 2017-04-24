cmake_minimum_required(VERSION 2.8)

project(DICOMProcessor)

list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package( ZLIB REQUIRED )
include_directories( ${ZLIB_INCLUDE_DIRS} )

add_executable(DICOMProcessor DICOMProcessor.cxx cnpy.cxx cnpy.h ImageProcessor.cxx ImageProcessor.h CommandLineProcessor.h optionparser.h ThreadedQueue.cpp ThreadedQueue.h)
target_link_libraries(DICOMProcessor ${ITK_LIBRARIES} ${ZLIB_LIBRARIES})