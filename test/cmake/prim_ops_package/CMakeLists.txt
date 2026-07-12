# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(executorch_prim_ops_package_test LANGUAGES CXX)

enable_testing()
find_package(executorch CONFIG REQUIRED)

foreach(_target prim_ops_lib executorch)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "ExecuTorch package does not export ${_target}")
  endif()
endforeach()

get_target_property(
  _compat_target executorch EXECUTORCH_WHOLE_ARCHIVE_TARGET
)
if(NOT _compat_target STREQUAL "prim_ops_lib")
  message(FATAL_ERROR "executorch does not forward to prim_ops_lib")
endif()
get_target_property(_compat_libraries executorch INTERFACE_LINK_LIBRARIES)
if(NOT "prim_ops_lib" IN_LIST _compat_libraries)
  message(FATAL_ERROR "executorch does not link prim_ops_lib")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../../tools/cmake/Utils.cmake")
executorch_target_link_options_shared_lib(executorch)

add_executable(prim_ops_consumer main.cpp)
target_link_libraries(prim_ops_consumer PRIVATE prim_ops_lib)
add_test(NAME prim_ops_consumer COMMAND prim_ops_consumer)

add_executable(executorch_compat_consumer main.cpp)
target_link_libraries(executorch_compat_consumer PRIVATE executorch)
add_test(NAME executorch_compat_consumer COMMAND executorch_compat_consumer)
