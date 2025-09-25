# C++ source files for IAPV library
set(IAPV_SOURCES
    common/math_utils.cpp
    planning/pathfinding.cpp
    decision_making/state_machine.cpp
    learning/neural_networks.cpp
    locomotion/steering_behaviors.cpp
    crowds/flocking.cpp
    communication/messaging.cpp
)

set(IAPV_HEADERS
    common/math_utils.hpp
    common/agent.hpp
    planning/pathfinding.hpp
    decision_making/state_machine.hpp
    learning/neural_networks.hpp
    locomotion/steering_behaviors.hpp
    crowds/flocking.hpp
    communication/messaging.hpp
    teamwork/coordination.hpp
)

# Create static library
add_library(iapv STATIC ${IAPV_SOURCES})

# Set include directories
target_include_directories(iapv PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link math library on Unix systems
if(UNIX)
    target_link_libraries(iapv m)
endif()

# Install headers
install(DIRECTORY . DESTINATION include/iapv
        FILES_MATCHING PATTERN "*.hpp")

# Install library
install(TARGETS iapv
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)