
file(GLOB_RECURSE SOURCES PARENT_SCOPE
	main.cpp
	../utils/*.cpp
)

set(SOURCES ${SOURCES} PARENT_SCOPE)

include_directories(
	"."
	"../external/stb"
	"../external/SDL/include"
	"../utils"
)

# Add source to this project's executable.
add_executable (${PROJECT_NAME}  ${SOURCES})

target_link_libraries(${PROJECT_NAME} SDL2::SDL2)

# Copy resources to build directory
add_custom_target(copy_assets ALL
  COMMENT "Copying assets to build directory"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/textures
          ${CMAKE_CURRENT_BINARY_DIR}/textures
  DEPENDS ${PROJECT_NAME}
)
