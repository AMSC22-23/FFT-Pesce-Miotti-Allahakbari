cmake_minimum_required(VERSION 3.0.0)
project(fft VERSION 0.1.0)

#Use c++20 standard
set(CMAKE_CXX_STANDARD 20)

#Set compiler flags for Release
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -O3 -Ofast -march=native -funroll-all-loops -DDOUBLE")

#Set flags for Debug
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -DDOUBLE")


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



