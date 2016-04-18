cmake_minimum_required(VERSION 2.8)
project( mosse )
find_package( OpenCV 3 REQUIRED )

#if(OpenCV_VERSION VERSION_LESS “3.0.0”)
#  message(FATAL_ERROR "ERROR: OpenCV 3.0.0”)
#endif()

add_executable( mosse main.cpp )
target_link_libraries( mosse ${OpenCV_LIBS} )