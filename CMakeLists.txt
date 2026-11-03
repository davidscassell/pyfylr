cmake_minimum_required(VERSION 3.15...3.27)
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT WIN32)
  set(CMAKE_CXX_COMPILER g++)
endif()
project(pyfylr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# A monorepo build uses core/ beside pyfylr. An sdist vendors that tree as
# _staged_core, so a wheel builds from the sdist with no sibling checkout.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../core/src/lta.cpp")
  set(LTA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../core")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_staged_core/src/lta.cpp")
  set(LTA_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_staged_core")
else()
  message(FATAL_ERROR
    "pyfylr needs the C++ core. Build from the monorepo, with core/ beside pyfylr, or from an sdist that contains _staged_core.")
endif()

pybind11_add_module(_core
  src/pyfylr/_core.cpp
  ${LTA_CORE_DIR}/src/lta.cpp
  ${LTA_CORE_DIR}/src/describe.cpp)

target_include_directories(_core PRIVATE
  ${LTA_CORE_DIR}/include
  ${LTA_CORE_DIR}/vendor)

find_package(Threads REQUIRED)
target_link_libraries(_core PRIVATE Threads::Threads)
target_compile_definitions(_core PRIVATE EIGEN_DONT_PARALLELIZE NDEBUG EIGEN_NO_DEBUG)
if(MSVC)
  # MSVC does not contract fused multiply-adds unless /fp:fast is requested.
  target_compile_options(_core PRIVATE /O2)
  target_compile_definitions(_core PRIVATE
    "LTA_COMPILE_FLAGS=\"/O2 -DEIGEN_DONT_PARALLELIZE -DNDEBUG -DEIGEN_NO_DEBUG\"")
else()
  target_compile_options(_core PRIVATE -fno-fast-math -ffp-contract=off -O2)
  target_compile_definitions(_core PRIVATE
    "LTA_COMPILE_FLAGS=\"-O2 -fno-fast-math -ffp-contract=off -DEIGEN_DONT_PARALLELIZE -DNDEBUG -DEIGEN_NO_DEBUG\"")
endif()

install(TARGETS _core DESTINATION pyfylr)
