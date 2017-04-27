cmake_minimum_required(VERSION 2.8)

project(DICOMProcessor)

list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package( ZLIB REQUIRED )
include_directories( ${ZLIB_INCLUDE_DIRS} )

add_executable(DICOMProcessor 
	CommandLineProcessor.h
	optionparser.h

	DICOMProcessor.cpp

	cnpy.h
	cnpy.cpp

	FullImageProcessor.h
	FullImageProcessor.cpp

	BinaryImageProcessor.h
	BinaryImageProcessor.cpp

	ThreadedQueue.h
	ThreadedQueue.cpp 

	tinyxml/tinystr.h
	tinyxml/tinystr.cpp 

	tinyxml/tinyxml.h
	tinyxml/tinyxml.cpp 

	tinyxml/tinyxmlerror.cpp
	tinyxml/tinyxmlparser.cpp 
)
					
target_link_libraries(DICOMProcessor ${ITK_LIBRARIES} ${ZLIB_LIBRARIES})