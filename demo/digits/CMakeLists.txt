
file(GLOB_RECURSE SOURCES PARENT_SCOPE
	main.cpp
	visualization/*.cpp
)

set(SOURCES ${SOURCES} PARENT_SCOPE)

add_subdirectory(external/SDL EXCLUDE_FROM_ALL)

find_package( OpenCV REQUIRED )

include_directories(
	"visualization"
	"tests"
	"external/stb"
	"external/SDL/include"
	${OpenCV_INCLUDE_DIRS}
)

# Add source to this project's executable.
add_executable (${PROJECT_NAME}  ${SOURCES})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} SDL2::SDL2)

# Copy resources to build directory
add_custom_target(copy_assets ALL
  COMMENT "Copying assets to build directory"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/dataset
          ${CMAKE_CURRENT_BINARY_DIR}/dataset
  DEPENDS ${PROJECT_NAME}
)
