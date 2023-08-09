
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("DeepPotato")

include_directories("src")

option(BUILD_XOR "Build XOR demo" OFF)
option(BUILD_IMAGE_COMPRESSION "Build Image Compression demo" OFF)
option(BUILD_DIGITS "Build Digits demo" OFF)

if(BUILD_XOR)
	add_subdirectory(demo/xor)
endif()

if(BUILD_IMAGE_COMPRESSION)
	add_subdirectory(demo/image_compression)
endif()

if(BUILD_DIGITS)
	add_subdirectory(demo/digits)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET DeepPotato PROPERTY CXX_STANDARD 20)
endif()

