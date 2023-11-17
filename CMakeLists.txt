cmake_minimum_required(VERSION 3.0.0)
project(fft VERSION 0.1.0)

#use c++20
set(CMAKE_CXX_STANDARD 20)

#Add all .cpp and .c files in src folder to compile list
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/*.c
    )

#Add all header files located include to headers list
include_directories(${PROJECT_SOURCE_DIR}/include)

#Setting target name
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME}
                      
                      )



