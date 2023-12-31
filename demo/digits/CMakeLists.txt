
file(GLOB_RECURSE SOURCES PARENT_SCOPE
	main.cpp
	../utils/*.cpp
)

set(SOURCES ${SOURCES} PARENT_SCOPE)

option(USE_OPENCV "Use OpenCV" OFF)

if(USE_OPENCV)
	find_package( OpenCV REQUIRED )
endif()

include_directories(
	"."
	"visualization"
	"Tests"
	"../external/stb"
	"../external/SDL/include"
	"../utils"
	${OpenCV_INCLUDE_DIRS}
)

# Add source to this project's executable.
add_executable (${PROJECT_NAME}  ${SOURCES})

if(USE_OPENCV)
	target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
endif()
target_link_libraries(${PROJECT_NAME} SDL2::SDL2)

# Copy resources to build directory
add_custom_target(copy_assets ALL
  COMMENT "Copying assets to build directory"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/dataset
          ${CMAKE_CURRENT_BINARY_DIR}/dataset
  DEPENDS ${PROJECT_NAME}
)
