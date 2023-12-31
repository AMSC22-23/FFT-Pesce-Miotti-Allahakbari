cmake_minimum_required(VERSION 3.0.0)
project(fft VERSION 0.1.0)
enable_language(CUDA)
#Use c++20
set(CMAKE_CXX_STANDARD 20)

#Set compiler flags for Release
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -Ofast -march=native -mtune=native -funroll-all-loops -flto -fno-signed-zeros -fno-trapping-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")  
#Set flags for Debug
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

find_package(CUDAToolkit REQUIRED)

#Add all .cpp and .c files in src folder to compile list
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                    ${PROJECT_SOURCE_DIR}/src/*.c
                    ${PROJECT_SOURCE_DIR}/src/*.cu

    )

#Add all header files located include to headers list
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})


#Set target name
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

#Add compile options
target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -fopenmp -DDOUBLE)

#Add link options
find_package(OpenMP)
target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX)
