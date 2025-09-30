# Demo applications
if(BUILD_EXAMPLES)
    # Simple console demo
    add_executable(console_demo console_demo.cpp)
    target_link_libraries(console_demo iapv)
    
    # Pathfinding demo
    add_executable(pathfinding_demo pathfinding_demo.cpp)
    target_link_libraries(pathfinding_demo iapv)
    
    # Flocking demo
    add_executable(flocking_demo flocking_demo.cpp)
    target_link_libraries(flocking_demo iapv)
    
    install(TARGETS console_demo pathfinding_demo flocking_demo
            RUNTIME DESTINATION bin)
endif()