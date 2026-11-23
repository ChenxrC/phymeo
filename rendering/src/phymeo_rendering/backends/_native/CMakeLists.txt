cmake_minimum_required(VERSION 3.20)
project(phymeo_native LANGUAGES CXX)
find_path(VULKAN_INCLUDE_DIR vulkan/vulkan.h
  HINTS "$ENV{VULKAN_SDK}/Include" "$ENV{VULKAN_SDK}/include")
find_library(VULKAN_LIBRARY NAMES vulkan-1 vulkan
  HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib")
find_program(GLSLANG_VALIDATOR NAMES glslang glslangValidator
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT VULKAN_INCLUDE_DIR OR NOT VULKAN_LIBRARY OR NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "Set VULKAN_INCLUDE_DIR, VULKAN_LIBRARY and GLSLANG_VALIDATOR; no SDK is downloaded")
endif()
add_library(phymeo_native SHARED renderer.cpp)
target_compile_features(phymeo_native PRIVATE cxx_std_17)
target_include_directories(phymeo_native PRIVATE "${VULKAN_INCLUDE_DIR}")
target_link_libraries(phymeo_native PRIVATE "${VULKAN_LIBRARY}")
if(MSVC)
  target_compile_options(phymeo_native PRIVATE /W4 /EHsc)
else()
  target_compile_options(phymeo_native PRIVATE -Wall -Wextra)
endif()
set_target_properties(phymeo_native PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(phymeo_native PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}")
endforeach()
foreach(program native resident resident_sky resident_rayquery resident_msaa_resolve)
  foreach(stage vert frag)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/${program}.${stage}")
    set(shader_options --target-env vulkan1.1)
    if(program STREQUAL "resident_rayquery")
      set(source "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident.${stage}")
      set(shader_options --target-env vulkan1.2 --glsl-version 460 -DPR_RESIDENT_RAY_QUERY=1)
    endif()
    set(output "${CMAKE_BINARY_DIR}/${program}.${stage}.spv")
    set(shader_dependencies "${source}")
    if((program STREQUAL "resident" OR program STREQUAL "resident_sky" OR program STREQUAL "resident_rayquery") AND stage STREQUAL "frag")
      list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_environment.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_display.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_msaa.glsl")
    endif()
    if((program STREQUAL "resident" OR program STREQUAL "resident_rayquery") AND stage STREQUAL "frag")
      list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_studio.glsl")
    endif()
    if(program STREQUAL "resident_rayquery" AND stage STREQUAL "frag")
      list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_rayquery.glsl")
    endif()
    add_custom_command(OUTPUT "${output}"
      COMMAND "${GLSLANG_VALIDATOR}" -V ${shader_options} "${source}" -o "${output}"
      DEPENDS ${shader_dependencies} VERBATIM)
    list(APPEND shader_outputs "${output}")
  endforeach()
endforeach()

# Explicit denser panel integration. Primary-visibility vertex programs are shared.
foreach(program resident resident_rayquery)
  foreach(grid 4 8)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident.frag")
    set(shader_options --target-env vulkan1.1 -DPR_STUDIO_AREA_GRID=${grid})
    set(shader_dependencies "${source}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_environment.glsl"
      "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_studio.glsl"
      "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_display.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_msaa.glsl")
    if(program STREQUAL "resident_rayquery")
      set(shader_options --target-env vulkan1.2 --glsl-version 460 -DPR_RESIDENT_RAY_QUERY=1 -DPR_STUDIO_AREA_GRID=${grid})
      list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/../_shaders/resident_rayquery.glsl")
    endif()
    set(output "${CMAKE_BINARY_DIR}/${program}_area${grid}.frag.spv")
    add_custom_command(OUTPUT "${output}"
      COMMAND "${GLSLANG_VALIDATOR}" -V ${shader_options} "${source}" -o "${output}"
      DEPENDS ${shader_dependencies} VERBATIM)
    list(APPEND shader_outputs "${output}")
  endforeach()
endforeach()

add_custom_target(native_shaders ALL DEPENDS ${shader_outputs})
add_dependencies(phymeo_native native_shaders)
