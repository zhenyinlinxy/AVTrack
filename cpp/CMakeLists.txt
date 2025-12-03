cmake_minimum_required(VERSION 3.10)
project(RobustTracker VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Option for CUDA support
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_TENSORRT "Enable TensorRT support" OFF)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

# Find ONNX Runtime
# Try to find onnxruntime using different methods
find_package(onnxruntime QUIET)

if(NOT onnxruntime_FOUND)
    # Try to find onnxruntime in common locations
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
            ${ONNXRUNTIME_ROOT}/include
            $ENV{ONNXRUNTIME_ROOT}/include
    )
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            /usr/lib
            /usr/local/lib
            ${ONNXRUNTIME_ROOT}/lib
            $ENV{ONNXRUNTIME_ROOT}/lib
    )
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(ONNXRUNTIME_FOUND TRUE)
        set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
        set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
        message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    else()
        message(WARNING "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT or install onnxruntime.")
        message(WARNING "You can install onnxruntime from: https://github.com/microsoft/onnxruntime/releases")
    endif()
else()
    message(STATUS "ONNX Runtime found via CMake config")
    set(ONNXRUNTIME_FOUND TRUE)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRUNTIME_FOUND AND ONNXRUNTIME_INCLUDE_DIRS)
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/robust_tracker.cpp
    src/reid_feature_bank.cpp
    src/osnet_infer.cpp
    src/utils.cpp
)

# Header files
set(HEADERS
    include/tracking_state.h
    include/reid_feature_bank.h
    include/osnet_infer.h
    include/utils.h
    include/robust_tracker.h
)

# Create executable
add_executable(robust_tracker ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(robust_tracker
    ${OpenCV_LIBS}
)

if(ONNXRUNTIME_FOUND)
    if(TARGET onnxruntime::onnxruntime)
        target_link_libraries(robust_tracker onnxruntime::onnxruntime)
    elseif(ONNXRUNTIME_LIBRARIES)
        target_link_libraries(robust_tracker ${ONNXRUNTIME_LIBRARIES})
    endif()
endif()

# CUDA support
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        target_compile_definitions(robust_tracker PRIVATE USE_CUDA)
        target_include_directories(robust_tracker PRIVATE ${CUDA_INCLUDE_DIRS})
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA not found, building without CUDA support")
    endif()
endif()

# TensorRT support
if(USE_TENSORRT)
    target_compile_definitions(robust_tracker PRIVATE USE_TENSORRT)
    message(STATUS "TensorRT support enabled")
endif()

# Create library target (optional, for use as a library)
add_library(robust_tracker_lib STATIC
    src/robust_tracker.cpp
    src/reid_feature_bank.cpp
    src/osnet_infer.cpp
    src/utils.cpp
)

target_include_directories(robust_tracker_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRUNTIME_FOUND AND ONNXRUNTIME_INCLUDE_DIRS)
    target_include_directories(robust_tracker_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
endif()

target_link_libraries(robust_tracker_lib
    ${OpenCV_LIBS}
)

if(ONNXRUNTIME_FOUND)
    if(TARGET onnxruntime::onnxruntime)
        target_link_libraries(robust_tracker_lib onnxruntime::onnxruntime)
    elseif(ONNXRUNTIME_LIBRARIES)
        target_link_libraries(robust_tracker_lib ${ONNXRUNTIME_LIBRARIES})
    endif()
endif()

# Install targets
install(TARGETS robust_tracker
    RUNTIME DESTINATION bin
)

install(TARGETS robust_tracker_lib
    ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS}
    DESTINATION include/robust_tracker
)

# Print summary
message(STATUS "")
message(STATUS "=== RobustTracker Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_FOUND}")
message(STATUS "CUDA support: ${USE_CUDA}")
message(STATUS "TensorRT support: ${USE_TENSORRT}")
message(STATUS "==========================================")
message(STATUS "")
