
file(GLOB_RECURSE SOURCES PARENT_SCOPE
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set(SOURCES ${SOURCES} PARENT_SCOPE)

# Add source to this project's executable.
add_executable (${PROJECT_NAME}  ${SOURCES})
