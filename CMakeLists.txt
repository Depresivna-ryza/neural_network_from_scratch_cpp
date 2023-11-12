cmake_minimum_required(VERSION 3.10)
project(
  neural_network
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_subdirectory(src)

add_executable(neural_network src/main.cpp src/miscellaneous.hpp src/neuralnetwork.hpp src/tensor.hpp)
add_executable(hyperparam_search src/hyperparam_search.cpp src/miscellaneous.hpp src/neuralnetwork.hpp src/tensor.hpp)

# target_link_libraries(neural_network <library_name>)

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# target_include_directories(neural_network PRIVATE src)

### TESTING ###

enable_testing()

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests src/tests.cpp src/miscellaneous.hpp src/neuralnetwork.hpp src/tensor.hpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)