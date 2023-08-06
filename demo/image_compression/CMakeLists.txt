
file(GLOB_RECURSE SOURCES PARENT_SCOPE
	main.cpp
	visualization/*.cpp
)

set(SOURCES ${SOURCES} PARENT_SCOPE)

add_subdirectory(external/SDL2-2.28.1 EXCLUDE_FROM_ALL)

include_directories(
	"visualization"
	"external/stb"
	"external/SDL2-2.28.1/include"
)

# Add source to this project's executable.
add_executable (${PROJECT_NAME}  ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2)

# Copy resources to build directory
add_custom_target(copy_assets ALL
  COMMENT "Copying assets to build directory"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/images
          ${CMAKE_CURRENT_BINARY_DIR}/images
  DEPENDS ${PROJECT_NAME}
)
